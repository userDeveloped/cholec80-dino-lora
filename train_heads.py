"""Heads on a frozen backbone: tool MLP and causal phase TCN, 3 seeds each, trained on the train videos 1-32.

1. Features: every frame goes once through the frozen backbone (skipped if <run>/features/<backbone>/ already holds
   all 80 videos). Each frame keeps its CLS token and the mean of its patch tokens.
2. Heads: trained on those saved features for a fixed number of epochs (chosen beforehand by a grid search on val; no selection here).
   The model at the end of training is saved. Val tool mAP and phase accuracy are printed for reference only.

  python train_heads.py --backbone baseline --frames <frames> --annotations <cholec80> --run <run> [--device cuda:0]
  python train_heads.py --backbone stage1 --checkpoint <run>/pretrain/best_checkpoint.pth --frames ... --run ...
Output: <run>/features/<backbone>/videoXX.pt and <run>/heads/<backbone>/{tools,phase}.pt (the 3 seeds in one file).
"""
import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

import data
import metrics
from models import CausalTCN, tool_head

# Head settings per backbone ("stage1" = our Stage 1 backbone, "baseline" = plain DINOv2)
SEEDS = [0, 1, 2]
TOOL_HEAD = dict(lr=3e-4, weight_decay=1e-2, batch=1024, epochs={"stage1": 51, "baseline": 52})
TCN = dict(lr=1e-3, weight_decay=1e-4, layers={"stage1": 12, "baseline": 8}, epochs={"stage1": 90, "baseline": 93})


def load_features(folder, videos):
    """Per-video feature dicts (videoXX.pt in folder), each with its video number added."""
    out = []
    for v in videos:
        d = torch.load(folder / f"video{v:02d}.pt")
        d["video"] = torch.full((len(d["frame"]),), v)
        out.append(d)
    return out


def tool_frames(per_video):
    """Frames with tool labels, concatenated: x = [cls, patch_mean] (1536-d), tool targets (float), video.
    (The last frame of each video has no tool label and is left out.)"""
    d = {k: torch.cat([p[k] for p in per_video]) for k in ("cls", "patch_mean", "tools", "video")}
    keep = d["tools"][:, 0] >= 0
    return torch.cat([d["cls"], d["patch_mean"]], dim=1)[keep], d["tools"][keep].float(), d["video"][keep]


def zscore_stats(x):
    """Feature normalization from train features: mean and std (+1e-6)."""
    return {"mean": x.mean(0), "std": x.std(0) + 1e-6}


def median_frequency_weights(phases):
    """Phase class weights median(f) / f_c from the train-frame frequencies (short phases weigh more)."""
    counts = torch.bincount(phases, minlength=len(data.PHASES)).float()
    freq = counts / counts.sum()
    return freq.median() / freq


def train_tool_head(x, y, seed, epochs):
    """Tool MLP on z-scored features x (on the GPU), plain BCE, AdamW, batch 1024, shuffled per epoch."""
    torch.manual_seed(seed)
    head = tool_head(x.shape[1]).to(x.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=TOOL_HEAD["lr"], weight_decay=TOOL_HEAD["weight_decay"])
    gen = torch.Generator(device=x.device).manual_seed(seed)
    head.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(x), generator=gen, device=x.device).split(TOOL_HEAD["batch"]):
            loss = F.binary_cross_entropy_with_logits(head(x[idx]), y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return head.eval()


def train_tcn(seqs, weights, layers, seed, epochs):
    """Causal TCN, one video per step in shuffled order, class-weighted cross-entropy summed over the stages."""
    torch.manual_seed(seed)
    model = CausalTCN(seqs[0][0].shape[1], layers).to(seqs[0][0].device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TCN["lr"], weight_decay=TCN["weight_decay"])
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        for i in torch.randperm(len(seqs), generator=gen).tolist():
            x, y = seqs[i]
            loss = sum(F.cross_entropy(o, y, weight=weights) for o in model(x))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval()


def train_tools(per_video, seed, epochs, device):
    """Tool head on the given train videos; z-score statistics from those videos (computed on the GPU)."""
    x, y, _ = tool_frames(per_video)
    x, y = x.to(device), y.to(device)
    norm = zscore_stats(x)
    head = train_tool_head((x - norm["mean"]) / norm["std"], y, seed, epochs)
    return {"seed": seed, "state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
            "normalization": {k: v.cpu() for k, v in norm.items()}}


def train_phase(per_video, layers, seed, epochs, device):
    """Causal TCN on the CLS sequences of the given train videos; z-score and class weights from those videos."""
    norm = zscore_stats(torch.cat([d["cls"] for d in per_video]))
    seqs = [(((d["cls"] - norm["mean"]) / norm["std"]).to(device), d["phase"].to(device)) for d in per_video]
    weights = median_frequency_weights(torch.cat([y for _, y in seqs]))
    model = train_tcn(seqs, weights, layers, seed, epochs)
    return {"seed": seed, "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "normalization": norm,
            "layers": layers}


@torch.no_grad()
def val_report(per_video, tool_saved, tcn_saved, device):
    """Val tool mAP (mean over seeds) and causal TCN phase accuracy (mean over videos and seeds), for reference."""
    x, y, _ = tool_frames(per_video)
    maps = []
    for saved in tool_saved:
        head = tool_head()
        head.load_state_dict(saved["state_dict"])
        norm = saved["normalization"]
        aps = metrics.tool_aps(head.to(device).eval()(((x - norm["mean"]) / norm["std"]).to(device)), y.to(device))
        maps.append(sum(aps) / len(aps))
    accs = []
    for saved in tcn_saved:
        model = CausalTCN(per_video[0]["cls"].shape[1], saved["layers"]).to(device)
        model.load_state_dict(saved["state_dict"])
        model.eval()
        norm = saved["normalization"]
        for d in per_video:
            pred = model(((d["cls"] - norm["mean"]) / norm["std"]).to(device))[-1].argmax(1).cpu()
            accs.append((pred == d["phase"]).float().mean().item())
    return {"val_tool_mAP": statistics.mean(maps), "val_tool_mAP_per_seed": maps,
            "val_phase_accuracy": statistics.mean(accs)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["stage1", "baseline"], required=True)
    parser.add_argument("--checkpoint", type=Path, help="Stage 1 checkpoint (best_checkpoint.pth); needed for stage1")
    parser.add_argument("--frames", type=Path, help="frames folder made by extract_frames.py (to extract features)")
    parser.add_argument("--annotations", type=Path,
                        help="Cholec80 folder containing phase_annotations/ and tool_annotations/ (to extract features)")
    parser.add_argument("--run", type=Path, required=True, help="run folder")
    parser.add_argument("--device", default="cuda:0", help="GPU, e.g. cuda:0 (nvidia-smi numbering)")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    b = args.backbone
    all_videos = data.TRAIN + data.VAL + data.TEST

    features_dir = args.run / "features" / b
    if not all((features_dir / f"video{v:02d}.pt").exists() for v in all_videos):
        assert args.frames and args.annotations, "--frames and --annotations are needed to extract the features"
        assert (b == "stage1") == (args.checkpoint is not None), "--checkpoint is needed for stage1 (and only for it)"
        print(f"extracting {b} features into {features_dir}", flush=True)
        backbone = data.load_backbone(device, args.checkpoint)
        data.extract_features(backbone, args.frames, args.annotations, all_videos, device, features_dir)

    out_dir = args.run / "heads" / b
    out_dir.mkdir(parents=True, exist_ok=True)
    train = load_features(features_dir, data.TRAIN)
    tool_saved, tcn_saved = [], []
    for seed in SEEDS:
        tool_saved.append(train_tools(train, seed, TOOL_HEAD["epochs"][b], device))
        tcn_saved.append(train_phase(train, TCN["layers"][b], seed, TCN["epochs"][b], device))
        print(f"seed {seed} done", flush=True)
    torch.save(tool_saved, out_dir / "tools.pt")
    torch.save(tcn_saved, out_dir / "phase.pt")
    print(json.dumps(val_report(load_features(features_dir, data.VAL), tool_saved, tcn_saved, device), indent=1))


if __name__ == "__main__":
    main()
