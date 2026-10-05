# Surgical phase and tool recognition on Cholec80: continued DINO pretraining + LoRA

Cholec80 has 80 videos of laparoscopic gallbladder removals. Each frame is labeled with one of 7 surgical phases and
with which of 7 tools are visible. This project asks: **does continuing the self-supervised training of a general
vision foundation model (DINOv2) on unlabeled surgical frames help it recognize phases and tools?**

1. **Stage 1, self-supervised:** DINOv2 ViT-B/14 (with registers) is trained further with the DINO objective on the
   train frames, **without labels**.
2. **Stage 2, with labels:** on top of a backbone, either plain DINOv2 ("baseline") or the Stage 1 backbone
   ("our DINO"), two ways of adding the labeled tasks:
   - **frozen backbone + heads:** the backbone is not changed; a tool head (MLP) and a phase model (causal temporal
     convolutional network, TCN) are trained on its features;
   - **LoRA + heads:** small low-rank adapters (LoRA) in the backbone's attention layers are trained together with
     the heads; all original backbone weights stay frozen. The phase TCN is then trained on the adapted features.

The phase model is **causal**: the prediction for a frame only uses that frame and earlier ones, as it would during a
live surgery.

## Results

Test set = videos 41-80, the standard Cholec80 test split, evaluated once with the final models. Tool mAP is the mean
of 3 seeds. Phase accuracy is per video, averaged over 3 seeds and then over the 40 videos; "relaxed" uses the
official Cholec80 evaluation (10 s tolerance at phase boundaries).

| model | tool mAP | phase accuracy | phase accuracy (relaxed) |
|---|---|---|---|
| plain DINOv2 + heads | 0.838 | 82.2% | 83.3% |
| our DINO + heads | 0.844 | 84.3% | 85.4% |
| plain DINOv2 + LoRA | 0.911 | 85.1% | 85.9% |
| **our DINO + LoRA** | **0.928** | **88.0%** | **88.7%** |

- **Self-supervised pretraining pays off most together with LoRA.** With frozen heads, our DINO is +0.007 tool mAP
  and +2.1 pt phase accuracy above plain DINOv2, which is within the variation between test videos (paired bootstrap
  over the 40 test videos in the original experiments). With LoRA it is +0.017 tool mAP and +2.9 pt, and the
  bootstrap interval excludes zero for both.
- **LoRA helps on both backbones**, most of all for thin instruments (scissors +0.14 / +0.26 AP, clipper +0.13 /
  +0.16), the irrigator and bipolar forceps. The grasper, visible in most frames, does not improve.

[`evaluate.ipynb`](evaluate.ipynb) shows the tables for train, val and test and three charts: AP per tool, phase over
time on three test videos (truth vs. each model), and phase confusion matrices. The
saved notebook contains the outputs of the final models; [`results.json`](results.json) has all
numbers.

## Data split

By video (= patient), never by frame: **train** videos 1-32, **val** 33-40, **test** 41-80.
- Stage 1 uses the train frames only, without labels.
- The val labels are used for three things only: the k-NN check that picks the Stage 1 checkpoint, the LoRA epoch,
  and choosing the head settings beforehand (a grid search on val).
- The test videos were evaluated once.

## Files

| file | what it does |
|---|---|
| `extract_frames.py` | videos -> frames at 1 fps (854x480 JPEGs), named by their frame number in the video |
| `data.py` | the split, the 7 phases and 7 tools, labels per frame, the frame dataset, the DINOv2 backbone (code pinned to one GitHub commit), feature extraction |
| `pretrain.py` | Stage 1: DINO continued pretraining (student/teacher, multi-crop, 65536 prototypes), k-NN check on val every 2 epochs, early stopping |
| `train_heads.py` | frozen backbone + tool head + causal phase TCN, 3 seeds |
| `lora.py` | LoRA (rank 8 on `qkv` of all 12 blocks) + tool head + phase head, epoch chosen on val; then the phase TCN on the LoRA features |
| `models.py` | tool head (MLP), causal multi-stage TCN, LoRA layer |
| `metrics.py` | tool AP/mAP, phase metrics (strict, and a port of the official relaxed Cholec80 evaluation), k-NN check |
| `evaluate.py` | results of the 4 models on train / val / test |
| `evaluate.ipynb` | runs `evaluate.py` and shows the tables and charts |
| `run.py` | trains everything from scratch in order, then evaluates |
| `results.json` | all numbers of the final models (train / val / test, every metric, AP per tool) |

All settings are at the top of the script that uses them.

## How to reproduce

**Requirements:** Linux, an NVIDIA GPU with driver >= 580 (CUDA 13.0), Python 3.12 and ffmpeg (tested with 6.1.1). The
Cholec80 dataset is available on request from [CAMMA](http://camma.u-strasbg.fr/datasets).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # exact versions used for the results (torch 2.14.0+cu130, ...)
```
On the first run, torch.hub downloads the DINOv2 code (pinned commit) and its pretrained weights (~350 MB) into
`$TORCH_HOME` (default `~/.cache/torch`).

**Everything with one command** (~4-5 h on one RTX PRO 6000 Blackwell, of which Stage 1 takes ~3 h):

```bash
python run.py --videos /path/to/cholec80/videos --annotations /path/to/cholec80 --run runs/run1 --device cuda:0
python run.py ... --dry-run      # only print the steps
```
Steps: frames -> Stage 1 -> frozen heads (both backbones) -> LoRA (both backbones) -> evaluation. Each finished step is
recorded in `runs/run1/done/`; running the same command again after an interruption continues with the next step.
Then open `evaluate.ipynb`, set `MODELS = Path("runs/run1")` and the other paths in its second cell, and run all
cells.

**Step by step**, the same as `run.py` does:

```bash
python extract_frames.py --videos /path/to/cholec80/videos --out runs/run1/frames
python pretrain.py   --frames runs/run1/frames --annotations /path/to/cholec80 --run runs/run1 --compile
python train_heads.py --backbone baseline --frames runs/run1/frames --annotations /path/to/cholec80 --run runs/run1
python train_heads.py --backbone stage1 --checkpoint runs/run1/pretrain/best_checkpoint.pth \
    --frames runs/run1/frames --annotations /path/to/cholec80 --run runs/run1
python lora.py --backbone baseline --frames runs/run1/frames --annotations /path/to/cholec80 --run runs/run1
python lora.py --backbone stage1 --checkpoint runs/run1/pretrain/best_checkpoint.pth \
    --frames runs/run1/frames --annotations /path/to/cholec80 --run runs/run1
```
`pretrain.py` and `lora.py` have a `--smoke` option (2 videos, a few hundred steps) to check that everything runs.

**What to expect:** the code, settings, package versions and DINOv2 commit are the ones that produced the results
above. GPU training is not fully deterministic, so a new run gives close but not bit-identical numbers (for the
phase TCN, identical runs varied by a few points of val accuracy). Features are computed in bf16, whose exact values
depend on how frames are batched; the scripts always batch them the same way.

## License and citation

Cholec80 is released under CC BY-NC-SA 4.0 (non-commercial use); please cite: A.P. Twinanda, S. Shehata, D. Mutter,
J. Marescaux, M. de Mathelin, N. Padoy, *EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos*,
IEEE Trans. on Medical Imaging 2016. DINOv2: M. Oquab et al., *DINOv2: Learning Robust Visual Features without
Supervision*, TMLR 2024; *Vision Transformers Need Registers*, T. Darcet et al., ICLR 2024.
