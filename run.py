"""Train everything from scratch, in order, then evaluate. Each step is one of the scripts in this folder.

  1. extract_frames.py              videos -> 1 fps frames (skipped with --frames)
  2. pretrain.py                    Stage 1: DINO continued pretraining on the train frames
  3. train_heads.py --backbone baseline / stage1   frozen backbone + tool head + phase TCN
  4. lora.py --backbone baseline / stage1          LoRA + heads, then the phase TCN on its features
  5. evaluate.py                    results on train / val / test -> <run>/results.json

evaluate.ipynb shows the results of a run folder (set MODELS to it).
A finished step is recorded in <run>/done/; running the same command again continues with the next step.
Takes ~4-5 h on one RTX PRO 6000 Blackwell (pretraining ~3 h).

  python run.py --videos <cholec80>/videos --annotations <cholec80> --run runs/run1 [--frames <frames>] [--device cuda:0]
  python run.py ... --dry-run      # only print the commands
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, help="folder with video01.mp4 ... video80.mp4 (not needed with --frames)")
    parser.add_argument("--frames", type=Path, help="already extracted frames (skips step 1)")
    parser.add_argument("--annotations", type=Path, required=True,
                        help="Cholec80 folder containing phase_annotations/ and tool_annotations/")
    parser.add_argument("--run", type=Path, required=True, help="folder for everything this run produces")
    parser.add_argument("--device", default="cuda:0", help="GPU, e.g. cuda:0 (nvidia-smi numbering)")
    parser.add_argument("--dry-run", action="store_true", help="only print the commands")
    args = parser.parse_args()
    assert args.frames or args.videos, "give --videos (to extract the frames) or --frames"
    run = args.run.resolve()
    frames = args.frames.resolve() if args.frames else run / "frames"
    checkpoint = run / "pretrain" / "best_checkpoint.pth"
    common = ["--frames", frames, "--annotations", args.annotations.resolve(), "--run", run, "--device", args.device]

    steps = [("pretrain", ["pretrain.py", *common, "--compile"]),
             ("heads_baseline", ["train_heads.py", "--backbone", "baseline", *common]),
             ("heads_stage1", ["train_heads.py", "--backbone", "stage1", "--checkpoint", checkpoint, *common]),
             ("lora_baseline", ["lora.py", "--backbone", "baseline", *common]),
             ("lora_stage1", ["lora.py", "--backbone", "stage1", "--checkpoint", checkpoint, *common])]
    if not args.frames:
        steps.insert(0, ("frames", ["extract_frames.py", "--videos", args.videos.resolve(), "--out", frames]))

    done = run / "done"
    for name, cmd in steps:
        cmd = [sys.executable, HERE / cmd[0], *cmd[1:]]
        if (done / name).exists():
            print(f"[{name}] already done")
            continue
        print(f"[{name}] " + " ".join(str(c) for c in cmd), flush=True)
        if not args.dry_run:
            subprocess.run([str(c) for c in cmd], cwd=HERE, check=True)
            done.mkdir(parents=True, exist_ok=True)
            (done / name).touch()

    print("[evaluate] train / val / test -> results.json (shown by evaluate.ipynb with MODELS = the run folder)")
    if not args.dry_run:
        sys.path.insert(0, str(HERE))
        import evaluate
        results = evaluate.evaluate(run, frames, args.annotations.resolve(), args.device)
        for name, r in results["test"].items():
            print(f"test  {name:14}  tool mAP {r['tool_mAP']:.3f}  phase accuracy {100 * r['phase_strict_accuracy']:.1f}%")


if __name__ == "__main__":
    main()
