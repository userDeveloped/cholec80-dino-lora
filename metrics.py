"""Metrics: tool AP / mAP, phase metrics (strict + official relaxed), k-NN check.

Phase metrics are per video (1 fps label sequences, phases 0-6) and then averaged over videos.
strict:  accuracy; per phase present in the ground truth: precision TP/(TP+FP) (0 if never predicted) and
         recall TP/(TP+FN); the video's P/R are the means over those phases.
relaxed: a line-by-line port of the official Cholec80 Evaluate.m (as distributed with TMRNet), quirks included:
         - 10 s boundary tolerance; wider tolerance for MATLAB phases 4-7;
         - the "early transition" mask is built from the LAST t frames of a segment but, through MATLAB logical
           indexing, applied to the FIRST t frames (Funke et al. 2023), so errors at the end of a phase are not forgiven;
         - precision NaN if a phase is never predicted and relaxed TP = 0 (left out of means), Inf -> 100% if TP > 0;
           values above 100% clipped (Main.m).
"""
import math

import torch
import torch.nn.functional as F

from data import PHASES


def average_precision(scores, targets):
    """AP of one class, sklearn's definition (precision/recall evaluated only at distinct score thresholds).
    NaN if the class has no positives."""
    order = scores.argsort(descending=True)
    scores, targets = scores[order], targets[order]
    tp, fp = targets.cumsum(0), (1 - targets).cumsum(0)
    last_of_tie = torch.cat([scores[1:] != scores[:-1], torch.tensor([True], device=scores.device)])
    tp, fp = tp[last_of_tie], fp[last_of_tie]
    precision, recall = tp / (tp + fp), tp / targets.sum()
    recall_step = torch.diff(recall, prepend=torch.zeros(1, device=recall.device))
    return (recall_step * precision).sum().item()


def tool_aps(scores, targets):
    """Per-tool AP (list of 7) of (frames, 7) scores and 0/1 targets."""
    return [average_precision(scores[:, c], targets[:, c]) for c in range(targets.shape[1])]


# ---------------------------------------------------------------- phase
def strict_metrics(gt, pred):
    """Accuracy and the mean precision / recall over the phases present in gt (+ per-phase lists, NaN if absent)."""
    per_phase = {"precision": [], "recall": []}
    for c in range(len(PHASES)):
        g, p = gt == c, pred == c
        if not g.any():
            for k in per_phase:
                per_phase[k].append(math.nan)
            continue
        tp, fp, fn = (g & p).sum().item(), (~g & p).sum().item(), (g & ~p).sum().item()
        per_phase["precision"].append(tp / (tp + fp) if tp + fp else 0.0)
        per_phase["recall"].append(tp / (tp + fn))
    out = {"accuracy": (gt == pred).float().mean().item()}
    for k, values in per_phase.items():
        present = [v for v in values if not math.isnan(v)]
        out[k] = sum(present) / len(present)
        out[f"{k}_per_phase"] = values
    return out


def _segments(mask):
    """(start, end) inclusive index pairs of the runs of True in a list (bwconncomp in 1-d)."""
    segs, start = [], None
    for i, m in enumerate(mask + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            segs.append((start, i - 1))
            start = None
    return segs


def official_relaxed(gt, pred, fps=1):
    """Port of Evaluate.m: per-phase precision / recall (NaN for phases absent from gt, not clipped) and the relaxed
    accuracy. (Evaluate.m also gives a Jaccard index; it is not used here. Its relaxed TP count over the union of
    predicted and ground-truth frames is what precision and recall are built on, so that part is kept.)"""
    gt = [int(v) + 1 for v in gt.tolist()]       # MATLAB phase ids 1-7
    pred = [int(v) + 1 for v in pred.tolist()]
    ori_t = 10 * fps
    diff = [p - g for p, g in zip(pred, gt)]
    updated = list(diff)

    def assign(cur, idx_mask, cond):
        # MATLAB cur(mask) = 0 with a mask shorter than cur: only the first len(mask) entries are indexed
        for i, v in enumerate(idx_mask):
            if cond(v):
                cur[i] = 0

    for phase in range(1, len(PHASES) + 1):
        for start, end in _segments([g == phase for g in gt]):
            cur = diff[start:end + 1]
            t = min(ori_t, len(cur))
            if phase in (4, 5):
                assign(cur, cur[:t], lambda v: v == -1)                   # late transition
                assign(cur, cur[len(cur) - t:], lambda v: v in (1, 2))    # early transition (applied to the first t)
            elif phase in (6, 7):
                assign(cur, cur[:t], lambda v: v in (-1, -2))
                assign(cur, cur[len(cur) - t:], lambda v: v in (1, 2))
            else:
                assign(cur, cur[:t], lambda v: v == -1)
                assign(cur, cur[len(cur) - t:], lambda v: v == 1)
            updated[start:end + 1] = cur

    precision, recall = [], []
    for phase in range(1, len(PHASES) + 1):
        gt_idx = {i for i, g in enumerate(gt) if g == phase}
        if not gt_idx:
            precision.append(math.nan), recall.append(math.nan)
            continue
        pred_idx = {i for i, p in enumerate(pred) if p == phase}
        tp = sum(updated[i] == 0 for i in gt_idx | pred_idx)  # relaxed TP over the union, as in Evaluate.m
        precision.append(tp / len(pred_idx) if pred_idx else (math.nan if tp == 0 else math.inf))
        recall.append(tp / len(gt_idx))
    return {"precision": precision, "recall": recall,
            "accuracy": sum(u == 0 for u in updated) / len(gt)}


def relaxed_metrics(gt, pred, fps=1):
    """Relaxed metrics of one video: accuracy and the nanmean over phases of the clipped P / R (+ per phase)."""
    r = official_relaxed(gt, pred, fps)
    out = {"accuracy": r["accuracy"]}
    for k in ("precision", "recall"):
        values = [v if math.isnan(v) else min(v, 1.0) for v in r[k]]  # Main.m: > 100% (incl. Inf) -> 100%
        present = [v for v in values if not math.isnan(v)]
        out[k] = sum(present) / len(present) if present else math.nan
        out[f"{k}_per_phase"] = values
    return out


def official_aggregate(per_video):
    """Main.m aggregation of relaxed_metrics() results: per phase the nanmean over videos, then the mean over phases."""
    out = {"accuracy": sum(v["accuracy"] for v in per_video) / len(per_video)}
    for k in ("precision", "recall"):
        means = []
        for c in range(len(PHASES)):
            vals = [v[f"{k}_per_phase"][c] for v in per_video if not math.isnan(v[f"{k}_per_phase"][c])]
            means.append(sum(vals) / len(vals) if vals else math.nan)
        present = [m for m in means if not math.isnan(m)]
        out[k] = sum(present) / len(present)
    return out


def phase_video_metrics(gt, pred):
    return {"strict": strict_metrics(gt, pred), "relaxed": relaxed_metrics(gt, pred)}


def knn_check(train, val, device, k=20, temperature=0.07):
    """Weighted k-NN (DINO protocol) on per-video feature dicts (data.extract_features output): the train frames are
    the memory bank, the val frames are classified. Only frames with tool labels; CLS features L2-normalized.
    Each of the k nearest train frames votes with weight exp(cos / T). Score = (phase accuracy + tool mAP) / 2."""
    sets = []
    for per_video in (train, val):
        d = {key: torch.cat([p[key] for p in per_video]) for key in ("cls", "phase", "tools")}
        keep = d["tools"][:, 0] >= 0
        sets.append((F.normalize(d["cls"][keep].to(device), dim=1), d["phase"][keep].to(device),
                     d["tools"][keep].float().to(device)))
    (tr_f, tr_p, tr_t), (va_f, va_p, va_t) = sets

    def vote(labels):
        scores = []
        for q in va_f.split(2048):
            sim, idx = (q @ tr_f.T).topk(k, dim=1)
            w = (sim / temperature).exp()
            scores.append((w.unsqueeze(-1) * labels[idx]).sum(1) / w.sum(1, keepdim=True))
        return torch.cat(scores)

    phase_acc = (vote(F.one_hot(tr_p, len(PHASES)).float()).argmax(1) == va_p).float().mean().item()
    aps = tool_aps(vote(tr_t), va_t)
    tool_map = sum(aps) / len(aps)
    return {"score": (phase_acc + tool_map) / 2, "phase_acc": phase_acc, "tool_map": tool_map}
