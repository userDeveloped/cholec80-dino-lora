"""Stage 1: continued DINO self-supervised pretraining of DINOv2 ViT-B/14 reg on the train frames (no labels).

Student and teacher (EMA of the student) see multi-crop views (2 x 224 global, 8 x 98 local); the student
learns to match the teacher's centered, sharpened output over 65536 prototypes (DINOHead, weight-normalized
last layer). Every 2 epochs the teacher backbone gets the k-NN check (val labels used only for this score);
the best is kept (best_checkpoint.pth), checkpoint.pth is the latest (resumed automatically). Early stop after
3 post-warmup checks without a gain of at least 0.002.

  python pretrain.py --frames <frames> --annotations <cholec80> --run <run> [--device cuda:0] [--compile] [--smoke]
Output: <run>/pretrain/ (train.log, checkpoint.pth, best_checkpoint.pth; knn.jsonl = every k-NN check,
knn_baseline.json = the k-NN check of plain DINOv2 before training).
--smoke: train videos 1-2, k-NN val video 33, 200 steps; only to check that everything runs.
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import DataLoader
from torchvision import transforms as T

import data
import metrics

# Stage 1 settings
P = dict(
    epochs=30, batch=128, n_local=8, workers=32, seed=0,
    backbone_lr=2.5e-5, head_lr=2.5e-4, warmup_epochs=5, weight_decay=0.04, clip_grad=3.0,
    teacher_temp=(0.04, 0.07), teacher_temp_epochs=10, momentum=(0.994, 1.0), freeze_last_layer_epochs=1,
    out_dim=65536, knn_every=2, patience=3, min_delta=0.002,
)


class DINOHead(nn.Module):
    """MLP 768 -> 2048 -> 2048 -> 256, L2-normalize, weight-normalized prototypes 256 -> 65536 (norm fixed to 1)."""

    def __init__(self, in_dim, out_dim=P["out_dim"], hidden=2048, bottleneck=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, bottleneck))
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        self.last_layer = weight_norm(nn.Linear(bottleneck, out_dim, bias=False))
        self.last_layer.parametrizations.weight.original0.data.fill_(1)
        self.last_layer.parametrizations.weight.original0.requires_grad = False

    def forward(self, x):
        return self.last_layer(F.normalize(self.mlp(x), dim=-1))


class DINO(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone, self.head = backbone, DINOHead(backbone.embed_dim)

    def forward(self, images):
        cls = self.backbone.forward_features(images)["x_norm_clstoken"]
        return cls, self.head(cls)


class DINOLoss(nn.Module):
    """Cross-entropy between the centered, sharpened teacher (global crops) and the student (all crops), skipping
    same-crop pairs. The center (EMA of teacher outputs) and the sharpening keep training from collapsing."""

    def __init__(self, out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp, self.center_momentum = student_temp, center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_out, teacher_out, teacher_temp):
        student = (student_out / self.student_temp).chunk(2 + P["n_local"])
        teacher = F.softmax((teacher_out - self.center) / teacher_temp, dim=-1).chunk(2)
        losses = [torch.sum(-t * F.log_softmax(s, dim=-1), dim=-1).mean()
                  for it, t in enumerate(teacher) for iv, s in enumerate(student) if iv != it]
        with torch.no_grad():
            m = self.center_momentum
            self.center = self.center * m + teacher_out.mean(0, keepdim=True) * (1 - m)
        return sum(losses) / len(losses)


class MultiCrop:
    """DINO augmentations: 2 global crops (224, scale 0.32-1) and n_local local crops (98, scale 0.05-0.32)."""

    def __init__(self):
        flip_color = T.Compose([T.RandomHorizontalFlip(), T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                                T.RandomGrayscale(p=0.2)])
        norm = T.Compose([T.ToTensor(), T.Normalize(data.MEAN, data.STD)])
        blur = lambda p: T.RandomApply([T.GaussianBlur(9, sigma=(0.1, 2.0))], p=p)
        crop = lambda size, scale: T.RandomResizedCrop(size, scale=scale, interpolation=T.InterpolationMode.BICUBIC)
        self.global1 = T.Compose([crop(224, (0.32, 1.0)), flip_color, blur(1.0), norm])
        self.global2 = T.Compose([crop(224, (0.32, 1.0)), flip_color, blur(0.1), T.RandomSolarize(128, p=0.2), norm])
        self.local = T.Compose([crop(98, (0.05, 0.32)), flip_color, blur(0.5), norm])

    def __call__(self, img):
        return [self.global1(img), self.global2(img)] + [self.local(img) for _ in range(P["n_local"])]


def cosine(start, end, step, total):
    return end + (start - end) * 0.5 * (1 + math.cos(math.pi * step / total))


def param_groups(model):
    """Backbone and head with their own learning rates; no weight decay on biases and norms."""
    groups = []
    for part, lr in ((model.backbone, P["backbone_lr"]), (model.head, P["head_lr"])):
        params = [p for p in part.parameters() if p.requires_grad]
        groups += [{"params": [p for p in params if p.ndim > 1], "base_lr": lr, "weight_decay": P["weight_decay"]},
                   {"params": [p for p in params if p.ndim <= 1], "base_lr": lr, "weight_decay": 0.0}]
    return groups


@torch.no_grad()
def collapse_monitors(teacher_out, teacher_cls, center, temp):
    """Teacher entropy per frame (near 0 = one-hot), entropy of the batch mean (low = every frame picks the same
    prototypes), std of the normalized CLS features (near 0 = identical features)."""
    p = F.softmax((teacher_out - center) / temp, dim=-1)
    return {"teacher_entropy": torch.special.entr(p).sum(-1).mean().item(),
            "batch_mean_entropy": torch.special.entr(p.mean(0)).sum().item(),
            "cls_std": F.normalize(teacher_cls.float(), dim=1).std(0).mean().item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True, help="frames folder made by extract_frames.py")
    parser.add_argument("--annotations", type=Path, required=True,
                        help="Cholec80 folder containing phase_annotations/ and tool_annotations/ (k-NN check)")
    parser.add_argument("--run", type=Path, required=True, help="run folder; the output goes to <run>/pretrain/")
    parser.add_argument("--device", default="cuda:0", help="GPU, e.g. cuda:0 (nvidia-smi numbering)")
    parser.add_argument("--compile", action="store_true", help="torch.compile (dynamic=False) the training step")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    train_videos, val_videos, max_steps = (data.TRAIN, data.VAL, None) if not args.smoke else ([1, 2], [33], 200)
    out_dir = args.run / "pretrain"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(P["seed"])

    def log(msg):
        print(msg, flush=True)
        with open(out_dir / "train.log", "a") as f:
            f.write(msg + "\n")

    loader = DataLoader(data.Frames(args.frames, args.annotations, train_videos, MultiCrop(), labels=False), batch_size=P["batch"], shuffle=True,
                        drop_last=True, num_workers=P["workers"], pin_memory=True, persistent_workers=True,
                        prefetch_factor=4, generator=torch.Generator().manual_seed(P["seed"]))
    student = DINO(torch.hub.load(data.DINOV2_REPO, data.DINOV2_MODEL, trust_repo=True)).to(device)
    teacher = copy.deepcopy(student).requires_grad_(False)
    dino_loss = DINOLoss(P["out_dim"]).to(device)
    optimizer = torch.optim.AdamW(param_groups(student), fused=True)
    # dynamic=False: one graph per crop size (avoids a compile issue in DINOv2's position-embedding resize)
    student_step = torch.compile(student, dynamic=False) if args.compile else student
    teacher_step = torch.compile(teacher, dynamic=False) if args.compile else teacher

    def knn_check(backbone):
        feats = data.extract_features(backbone, args.frames, args.annotations, train_videos + val_videos, device)
        return metrics.knn_check([feats[v] for v in train_videos], [feats[v] for v in val_videos], device)

    if not (out_dir / "knn_baseline.json").exists():  # the starting point: plain DINOv2, before any training
        (out_dir / "knn_baseline.json").write_text(json.dumps(knn_check(teacher.backbone), indent=1))

    start, best_score, bad_checks = 0, float("-inf"), 0
    if (out_dir / "checkpoint.pth").exists():
        ckpt = torch.load(out_dir / "checkpoint.pth", map_location=device)
        for obj, key in ((student, "student"), (teacher, "teacher"), (optimizer, "optimizer"), (dino_loss, "dino_loss")):
            obj.load_state_dict(ckpt[key])
        start, best_score, bad_checks = ckpt["epoch"] + 1, ckpt["best_score"], ckpt["bad_checks"]
        log(f"resumed after epoch {ckpt['epoch']}")
    steps, total = len(loader), P["epochs"] * len(loader)
    warmup = P["warmup_epochs"] * steps
    log(f"{len(loader.dataset)} frames, {steps} steps/epoch, {P['epochs']} epochs, compile {args.compile}")

    for epoch in range(start, P["epochs"]):
        t0 = time.perf_counter()
        for i, images in enumerate(loader):
            step = epoch * steps + i
            lr_factor = step / warmup if step < warmup else cosine(1.0, 0.0, step - warmup, total - warmup)
            for g in optimizer.param_groups:
                g["lr"] = g["base_lr"] * lr_factor
            temp = P["teacher_temp"][0] + (P["teacher_temp"][1] - P["teacher_temp"][0]) * min(1.0, epoch / P["teacher_temp_epochs"])
            momentum = cosine(*P["momentum"], step, total)

            images = [im.to(device, non_blocking=True) for im in images]
            global_crops, local_crops = torch.cat(images[:2]), torch.cat(images[2:])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    teacher_cls, teacher_out = teacher_step(global_crops)
                student_out = torch.cat([student_step(global_crops)[1], student_step(local_crops)[1]])
            center = dino_loss.center  # value used by this step's loss (the loss replaces it afterwards)
            loss = dino_loss(student_out.float(), teacher_out.float(), temp)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), P["clip_grad"])
            if epoch < P["freeze_last_layer_epochs"]:
                for p in student.head.last_layer.parameters():
                    p.grad = None
            optimizer.step()
            with torch.no_grad():  # teacher = EMA of the student
                for t, s in zip(teacher.parameters(), student.parameters()):
                    t.lerp_(s, 1 - momentum)
            if i % 50 == 0:
                mon = collapse_monitors(teacher_out.float(), teacher_cls, center, temp)
                log(f"epoch {epoch} step {i}/{steps} loss {loss.item():.4f} " + " ".join(f"{k} {v:.3f}" for k, v in mon.items())
                    + f" lr {optimizer.param_groups[0]['lr']:.2e} temp {temp:.3f} m {momentum:.4f} [{time.perf_counter() - t0:.0f}s]")
            if max_steps and step + 1 >= max_steps:
                break

        is_best = stop = False
        if (epoch + 1) % P["knn_every"] == 0 or epoch + 1 == P["epochs"]:
            knn = knn_check(teacher.backbone)
            if knn["score"] >= best_score + P["min_delta"]:
                best_score, bad_checks, is_best = knn["score"], 0, True
            elif epoch + 1 > P["warmup_epochs"]:  # checks during warmup never count toward stopping
                bad_checks += 1
            stop = bad_checks >= P["patience"]
            log(f"epoch {epoch} k-NN {json.dumps(knn)}; best {best_score:.4f}, {bad_checks}/{P['patience']} bad checks")
            with open(out_dir / "knn.jsonl", "a") as f:
                f.write(json.dumps({"epoch": epoch, "best": is_best, **knn}) + "\n")
        state = {"student": student.state_dict(), "teacher": teacher.state_dict(), "optimizer": optimizer.state_dict(),
                 "dino_loss": dino_loss.state_dict(), "epoch": epoch, "best_score": best_score, "bad_checks": bad_checks}
        torch.save(state, out_dir / "checkpoint.pth")
        if is_best:  # the teacher backbone of best_checkpoint.pth is the Stage 1 backbone
            torch.save(state, out_dir / "best_checkpoint.pth")
        if stop or (max_steps and (epoch + 1) * steps >= max_steps):
            log(f"stopped after epoch {epoch}")
            break


if __name__ == "__main__":
    main()
