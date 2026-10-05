"""LoRA on a frozen backbone + jointly trained tool and phase heads, then a causal TCN on the LoRA features.

- every original backbone weight frozen (SHA-256 over all of them checked before and after training);
- LoRA rank 8, alpha 16 on attn.qkv of all 12 blocks;
- tool head: MLP on z-scored cls+patchmean (BCE; frames without tool labels count for phase only);
  phase head: linear on z-scored cls (median-frequency class-weighted CE); loss = sum; z-score = this
  backbone's frozen train features from train_heads.py (<run>/features/<backbone>/, so run train_heads.py first);
- 224x392 whole frames with horizontal flip and light color jitter; bf16; AdamW (LoRA lr 1e-4 wd 0,
  heads lr 1e-3 wd 1e-2), batch 128, 10 epochs; the epoch is chosen on val by
  (tool mAP + mean per-video per-frame phase accuracy) / 2 and THAT epoch's model is saved;
- features of all 80 videos from the saved model, then the causal TCN (3 seeds) with the backbone's settings
  from train_heads.py.

  python lora.py --backbone baseline --frames <frames> --annotations <cholec80> --run <run> [--device cuda:0]
  python lora.py --backbone stage1 --checkpoint <run>/pretrain/best_checkpoint.pth --frames ... --run ...
Output: <run>/lora_<backbone>/ (lora.pt, phase.pt with the 3 TCN seeds, summary.json), <run>/features/<backbone>_lora/.
--smoke: train videos 1-2, val video 33, TCN 3 epochs; only to check that everything runs.
"""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

import data
import metrics
import train_heads
from models import CausalTCN, LoRALinear, tool_head

# LoRA settings
LORA = dict(rank=8, alpha=16, lora_lr=1e-4, head_lr=1e-3, head_wd=1e-2, batch=128, epochs=10, workers=16, seed=0,
            color_jitter=(0.2, 0.2, 0.1, 0.02))
TRAIN_TF = T.Compose([T.Resize(data.INPUT_SIZE, interpolation=T.InterpolationMode.BICUBIC), T.RandomHorizontalFlip(),
                      T.ColorJitter(*LORA["color_jitter"]), T.ToTensor(), T.Normalize(data.MEAN, data.STD)])
IS_LORA = (".A", ".B")


def frozen_checksum(backbone):
    """SHA-256 over every original (non-LoRA) weight in name order (names as before wrapping qkv)."""
    h = hashlib.sha256()
    for n, t in sorted(backbone.state_dict().items()):
        if not n.endswith(IS_LORA):
            h.update(n.replace(".qkv.base.", ".qkv.").encode())
            h.update(t.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


class JointModel(nn.Module):
    """Frozen backbone with LoRA on every attn.qkv + tool head (MLP on z-scored cls+patchmean) + phase head (linear on z-scored cls)."""

    def __init__(self, backbone, norm):
        super().__init__()
        for block in backbone.blocks:
            block.attn.qkv = LoRALinear(block.attn.qkv, LORA["rank"], LORA["alpha"])
        self.backbone = backbone
        self.register_buffer("mean", norm["mean"])
        self.register_buffer("std", norm["std"])
        self.tool_head = tool_head(1536)
        self.phase_head = nn.Linear(768, len(data.PHASES))

    def features(self, x):
        return torch.cat(data.cls_and_patch_mean(self.backbone, x), dim=1)

    def forward(self, x):
        z = (self.features(x) - self.mean) / self.std
        return self.tool_head(z), self.phase_head(z[:, :768])

    def trainable_state(self):
        """LoRA weights, heads and normalization (the frozen weights come from the backbone checkpoint / hub)."""
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                if k.endswith(IS_LORA) or k.startswith(("tool_head", "phase_head", "mean", "std"))}


def build(args, device):
    """Joint model on the frozen backbone; returns it and the frozen-weight checksum (taken before adding LoRA)."""
    backbone = data.load_backbone(device, args.checkpoint)
    checksum = frozen_checksum(backbone)
    frozen_train = train_heads.load_features(args.run / "features" / args.backbone, data.TRAIN)
    x, _, _ = train_heads.tool_frames(frozen_train)
    model = JointModel(backbone, train_heads.zscore_stats(x.to(device))).to(device)
    return model, checksum


@torch.no_grad()
def validate(model, args, videos, device):
    """Val tool mAP, mean per-video per-frame phase accuracy, and the selection score (their mean)."""
    model.eval()
    loader = DataLoader(data.Frames(args.frames, args.annotations, videos), batch_size=256,
                        num_workers=LORA["workers"], pin_memory=True)
    ts, tt, pp, py, vid = [], [], [], [], []
    for x, p, t, v in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tool_logits, phase_logits = model(x.to(device, non_blocking=True))
        ts.append(tool_logits.float()), tt.append(t.to(device)), pp.append(phase_logits.argmax(1).cpu()), py.append(p), vid.append(v)
    ts, tt, pp, py, vid = map(torch.cat, (ts, tt, pp, py, vid))
    keep = tt[:, 0] >= 0
    aps = metrics.tool_aps(ts[keep], tt[keep])
    acc = statistics.mean((pp[vid == v] == py[vid == v]).float().mean().item() for v in videos)
    return {"tool_mAP": sum(aps) / len(aps), "phase_frame_acc": acc, "score": (sum(aps) / len(aps) + acc) / 2}


def train(args, train_videos, val_videos, out_dir, device):
    """Trains LoRA + heads; saves the model of the epoch with the best val score as lora.pt."""
    torch.manual_seed(LORA["seed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model, checksum = build(args, device)
    lora_params = [p for n, p in model.backbone.named_parameters() if n.endswith(IS_LORA)]
    head_params = list(model.tool_head.parameters()) + list(model.phase_head.parameters())
    optimizer = torch.optim.AdamW([{"params": lora_params, "lr": LORA["lora_lr"], "weight_decay": 0.0},
                                   {"params": head_params, "lr": LORA["head_lr"], "weight_decay": LORA["head_wd"]}])
    loader = DataLoader(data.Frames(args.frames, args.annotations, train_videos, TRAIN_TF), batch_size=LORA["batch"],
                        shuffle=True, drop_last=True, num_workers=LORA["workers"], pin_memory=True,
                        persistent_workers=True, generator=torch.Generator().manual_seed(LORA["seed"]))
    phases = torch.cat([data.video_labels(args.frames, args.annotations, v)["phase"] for v in train_videos])
    phase_w = train_heads.median_frequency_weights(phases).to(device)
    print(f"{len(loader.dataset)} train frames, {len(loader)} steps/epoch, frozen checksum {checksum[:16]}", flush=True)
    best, history = None, []
    for epoch in range(1, LORA["epochs"] + 1):
        model.train()
        t0 = time.perf_counter()
        for x, p, t, _ in loader:
            x, p, t = x.to(device, non_blocking=True), p.to(device, non_blocking=True), t.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tool_logits, phase_logits = model(x)
            keep = t[:, 0] >= 0
            loss = (F.binary_cross_entropy_with_logits(tool_logits.float()[keep], t[keep])
                    + F.cross_entropy(phase_logits.float(), p, weight=phase_w))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        val = validate(model, args, val_videos, device)
        history.append({"epoch": epoch, **val, "train_s": time.perf_counter() - t0})
        if best is None or val["score"] > best["score"]:  # save the model of THIS epoch
            best = {"epoch": epoch, **val}
            torch.save({"epoch": epoch, "state": model.trainable_state(), "val": val, "frozen_checksum": checksum},
                       out_dir / "lora.pt")
        print(f"epoch {epoch}: loss {loss.item():.4f}, val {val}{'  -> saved' if best['epoch'] == epoch else ''}", flush=True)
    summary = {"best": best, "history": history, "frozen_checksum_before": checksum,
               "frozen_checksum_after": frozen_checksum(model.backbone)}
    summary["frozen_unchanged"] = summary["frozen_checksum_before"] == summary["frozen_checksum_after"]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"selected epoch {best['epoch']}; frozen weights unchanged: {summary['frozen_unchanged']}")
    return summary


def load_trained(args, path, device):
    """The saved (selected-epoch) model; checks that the frozen weights are those it was trained on."""
    model, checksum = build(args, device)
    saved = torch.load(path)
    assert saved["frozen_checksum"] == checksum
    missing = model.load_state_dict(saved["state"], strict=False)
    assert not missing.unexpected_keys and not any(k.endswith(IS_LORA) or "head" in k for k in missing.missing_keys)
    return model.eval(), saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["stage1", "baseline"], required=True)
    parser.add_argument("--checkpoint", type=Path, help="Stage 1 checkpoint (best_checkpoint.pth); needed for stage1")
    parser.add_argument("--frames", type=Path, required=True, help="frames folder made by extract_frames.py")
    parser.add_argument("--annotations", type=Path, required=True,
                        help="Cholec80 folder containing phase_annotations/ and tool_annotations/")
    parser.add_argument("--run", type=Path, required=True, help="run folder (with <run>/features/<backbone>/ from train_heads.py)")
    parser.add_argument("--device", default="cuda:0", help="GPU, e.g. cuda:0 (nvidia-smi numbering)")
    parser.add_argument("--smoke", action="store_true", help="2 train videos, 1 val video, TCN 3 epochs")
    args = parser.parse_args()
    assert (args.backbone == "stage1") == (args.checkpoint is not None), "--checkpoint is needed for stage1 (and only for it)"
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    b = args.backbone
    out_dir, features_dir = args.run / f"lora_{b}", args.run / "features" / f"{b}_lora"
    if args.smoke:
        train_videos, val_videos, all_videos, tcn_epochs = [1, 2], [33], [1, 2, 33], 3
    else:
        train_videos, val_videos, all_videos = data.TRAIN, data.VAL, data.TRAIN + data.VAL + data.TEST
        tcn_epochs = train_heads.TCN["epochs"][b]
    summary = train(args, train_videos, val_videos, out_dir, device)

    model, saved = load_trained(args, out_dir / "lora.pt", device)
    print(f"reloaded epoch {saved['epoch']}: val {validate(model, args, val_videos, device)} (saved: {saved['val']})")
    assert saved["epoch"] == summary["best"]["epoch"]
    data.extract_features(model.backbone, args.frames, args.annotations, all_videos, device, features_dir, per_video=True)

    # causal TCN on the LoRA features, with this backbone's settings from train_heads.py
    tcn_train = train_heads.load_features(features_dir, train_videos)
    tcn_val = train_heads.load_features(features_dir, val_videos)
    tcns, accs = [], []
    for seed in train_heads.SEEDS:
        saved_tcn = train_heads.train_phase(tcn_train, train_heads.TCN["layers"][b], seed, tcn_epochs, device)
        tcns.append(saved_tcn)
        tcn = CausalTCN(768, saved_tcn["layers"]).to(device)
        tcn.load_state_dict(saved_tcn["state_dict"])
        tcn.eval()
        norm = saved_tcn["normalization"]
        with torch.no_grad():
            accs.append(statistics.mean(
                (tcn(((d["cls"] - norm["mean"]) / norm["std"]).to(device))[-1].argmax(1).cpu() == d["phase"]).float().mean().item()
                for d in tcn_val))
    torch.save(tcns, out_dir / "phase.pt")
    print(f"TCN on LoRA features, val phase accuracy per seed: {accs}")


if __name__ == "__main__":
    main()
