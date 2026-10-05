"""Results of the 4 final models on the train, val and test videos, from a folder of trained models.

The folder is a run folder of run.py (or any folder with the same layout):
  pretrain/best_checkpoint.pth, heads/{baseline,stage1}/{tools,phase}.pt, lora_{baseline,stage1}/{lora,phase}.pt
Features of all 80 videos are extracted once per model into <models>/features/ (skipped if already there; always
all 80 videos at once, batched as when the models were trained, so that the bf16 features are exactly reproducible).

Models: baseline / stage1 = frozen plain DINOv2 / Stage 1 backbone + tool head and causal phase TCN (3 seeds each);
baseline_lora / stage1_lora = the same backbone + LoRA + tool head, and a causal TCN on the LoRA features (3 seeds).
Tool mAP is the mean over seeds; phase metrics are per video, averaged over seeds, then over videos.

Used from evaluate.ipynb: results = evaluate(models, frames, annotations, device); also saved as <models>/results.json.
"""
import hashlib
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch

import data
import lora
import metrics
from models import CausalTCN, tool_head

MODELS = ["baseline", "stage1", "baseline_lora", "stage1_lora"]
SPLITS = {"train": data.TRAIN, "val": data.VAL, "test": data.TEST}
PHASE_KEYS = [(mode, m) for mode in ("strict", "relaxed") for m in ("accuracy", "precision", "recall")]


def check_sums(models):
    """Every file listed in <models>/SHA256SUMS (if there) unchanged."""
    sums = models / "SHA256SUMS"
    if sums.exists():
        for line in sums.read_text().splitlines():
            digest, rel = line.split("  ")
            assert hashlib.sha256((models / rel).read_bytes()).hexdigest() == digest, f"{rel} does not match SHA256SUMS"
        print("all files match SHA256SUMS", flush=True)


def features(name, args, device):
    """Folder with the features of all 80 videos for one model, extracted first if missing."""
    folder = args.models / "features" / name
    all_videos = data.TRAIN + data.VAL + data.TEST
    if not all((folder / f"video{v:02d}.pt").exists() for v in all_videos):
        b = name.removesuffix("_lora")
        checkpoint = args.models / "pretrain" / "best_checkpoint.pth" if b == "stage1" else None
        if name.endswith("_lora"):  # the backbone with its trained LoRA adapters
            ns = SimpleNamespace(backbone=b, checkpoint=checkpoint, run=args.models)
            backbone = lora.load_trained(ns, args.models / f"lora_{b}" / "lora.pt", device)[0].backbone
        else:
            backbone = data.load_backbone(device, checkpoint)
        print(f"extracting {name} features into {folder}", flush=True)
        data.extract_features(backbone, args.frames, args.annotations, all_videos, device, folder,
                              per_video=name.endswith("_lora"))
    return folder


def load_features(folder, videos):
    """Per-video feature dicts {frame, phase, tools, cls, patch_mean}."""
    return [torch.load(folder / f"video{v:02d}.pt") for v in videos]


def tool_heads(name, models):
    """The tool head(s) of a model as [(state_dict, normalization)]: 3 seeds, or the one trained with LoRA."""
    if name.endswith("_lora"):
        state = torch.load(models / f"lora_{name.removesuffix('_lora')}" / "lora.pt")["state"]
        return [({k.removeprefix("tool_head."): v for k, v in state.items() if k.startswith("tool_head.")},
                 {"mean": state["mean"].cpu(), "std": state["std"].cpu()})]
    return [(s["state_dict"], s["normalization"]) for s in torch.load(models / "heads" / name / "tools.pt")]


def phase_tcns(name, models):
    """The 3 causal TCNs (one per seed) of a model."""
    folder = models / (f"lora_{name.removesuffix('_lora')}" if name.endswith("_lora") else f"heads/{name}")
    return torch.load(folder / "phase.pt")


@torch.no_grad()
def evaluate_model(per_video, tools, tcns, device):
    """Per-seed APs of each tool head; per-seed, per-video phase metrics of each TCN."""
    d = {k: torch.cat([p[k] for p in per_video]) for k in ("cls", "patch_mean", "tools")}
    keep = d["tools"][:, 0] >= 0  # the last frame of each video has no tool label
    x, targets = torch.cat([d["cls"], d["patch_mean"]], 1)[keep], d["tools"][keep].float().to(device)
    ap_per_seed = []
    for state_dict, norm in tools:
        head = tool_head()
        head.load_state_dict(state_dict)
        ap_per_seed.append(metrics.tool_aps(head.to(device).eval()(((x - norm["mean"]) / norm["std"]).to(device)), targets))
    phase = []
    for saved in tcns:
        preds = predict_phase(saved, per_video, device)
        phase.append([metrics.phase_video_metrics(p["phase"], pr) for p, pr in zip(per_video, preds)])
    return {"ap_per_seed": ap_per_seed, "phase": phase}


@torch.no_grad()
def predict_phase(saved, per_video, device):
    """Phase per frame (0-6) of each video from one saved causal TCN (its last stage)."""
    model = CausalTCN(per_video[0]["cls"].shape[1], saved["layers"]).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    norm = saved["normalization"]
    return [model(((p["cls"] - norm["mean"]) / norm["std"]).to(device))[-1].argmax(1).cpu() for p in per_video]


def phase_predictions(models, name, videos, device, seed=0):
    """{video: (true phases, predicted phases)} of one model (one TCN seed), from the features evaluate() saved."""
    models = Path(models)
    per_video = load_features(models / "features" / name, videos)
    preds = predict_phase(phase_tcns(name, models)[seed], per_video, torch.device(device))
    return {v: (p["phase"], pr) for v, p, pr in zip(videos, per_video, preds)}


def per_video_phase(r, mode, m):
    """Per-video values of one phase metric, averaged over the seeds."""
    return [statistics.mean(seed[i][mode][m] for seed in r["phase"]) for i in range(len(r["phase"][0]))]


def summary(r):
    """Tool mAP (mean over seeds) and per-tool AP; phase metrics as the mean over videos of the seed-averaged values;
    the relaxed metrics also with the official Main.m aggregation (mean over seeds)."""
    maps = [sum(a) / len(a) for a in r["ap_per_seed"]]
    out = {"tool_mAP": statistics.mean(maps), "tool_mAP_per_seed": maps,
           "tool_ap": {t: statistics.mean(a[c] for a in r["ap_per_seed"]) for c, t in enumerate(data.TOOLS)}}
    for mode, m in PHASE_KEYS:
        out[f"phase_{mode}_{m}"] = statistics.mean(per_video_phase(r, mode, m))
    official = [metrics.official_aggregate([v["relaxed"] for v in seed]) for seed in r["phase"]]
    out["phase_relaxed_official"] = {k: statistics.mean(o[k] for o in official) for k in official[0]}
    return out


def evaluate(models, frames, annotations, device, splits=("train", "val", "test")):
    """{split: {model: summary}} for the given splits; also written to <models>/results.json."""
    models = Path(models)
    args = SimpleNamespace(models=models, frames=Path(frames), annotations=Path(annotations))
    device = torch.device(device)
    torch.cuda.set_device(device)
    check_sums(models)
    folders = {name: features(name, args, device) for name in MODELS}  # the frozen ones first: LoRA needs them
    results = {}
    for split in splits:
        results[split] = {name: summary(evaluate_model(load_features(folders[name], SPLITS[split]),
                                                       tool_heads(name, models), phase_tcns(name, models), device))
                          for name in MODELS}
        print(f"{split} done", flush=True)
    (models / "results.json").write_text(json.dumps(results, indent=1))
    return results
