"""Extract Cholec80 frames at 1 fps (every 25th frame) as 854x480 JPEGs with ffmpeg.

  python extract_frames.py --videos /path/to/cholec80/videos --out /path/to/frames [--only 1 2]
"""
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STEP = 25          # the videos are 25 fps: keep every 25th frame = 1 fps
SIZE = "854:480"   # most videos are 854x480; downscale the 1080p ones (78-80) so all frames share one size
THREADS = 4        # ffmpeg threads per video


def extract(video, out_dir):
    out = out_dir / video.stem
    out.mkdir(parents=True, exist_ok=True)
   
    subprocess.run(
        ["ffmpeg", "-v", "error", "-threads", str(THREADS), "-i", str(video),
         "-vf", f"select='not(mod(n,{STEP}))',scale={SIZE}", "-fps_mode", "passthrough",
         "-q:v", "2", "-start_number", "0", str(out / "tmp_%06d.jpg")],
        check=True)

    files = sorted(out.glob("tmp_*.jpg"))

    for k, f in enumerate(files):   # renaming to frame numbers
        f.rename(out / f"{k * STEP:06d}.jpg")
    print(f"{video.stem}: {len(files)} frames", flush=True)
    return len(files)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, required=True, help="folder with video01.mp4 ... video80.mp4")
    parser.add_argument("--out", type=Path, required=True, help="output folder for the frames")
    parser.add_argument("--only", type=int, nargs="+", help="extract only these video numbers (for a quick test)")
    parser.add_argument("--jobs", type=int, default=12, help="videos extracted in parallel")
    args = parser.parse_args()
    
    videos = [args.videos / f"video{v:02d}.mp4" for v in (args.only or range(1, 81))]
    videos.sort(key=lambda v: v.stat().st_size, reverse=True)  # sort the videos, the longest first, so the pool finishes evenly


    # Extract up to args.jobs videos in parallel; each worker picks up the next video when it finishes one.
    with ThreadPoolExecutor(args.jobs) as pool:
        total = sum(pool.map(lambda v: extract(v, args.out), videos))  # The per-video frame counts are summed only to print a final total as a debugging check.
    print(f"done: {total} frames from {len(videos)} videos")


if __name__ == "__main__":
    main()
