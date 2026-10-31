#!/usr/bin/env python3
"""Render the preprocessing windows into web clips, and index them for `preprocessing.html`.

Runs where the preprocessing output lives, not in the browser. For each recording it finds the
manipulation window's overlay frames — the rectified upright RGB with the tracked object
keypoints and hand landmarks drawn on — encodes them into one mp4, and writes an index with the
frame counts and the objects each window tracks.

The object names come from the preprocessing job's own log, where the stage that writes the
reference masks announces what it drew. That is the only place the choice is recorded per
recording, and it is read, never guessed: a recording whose log is missing gets an empty list
rather than a plausible name.

Usage:
    python3 scripts/hot3d_preprocess_clips.py <runs_dir> <seqs.txt> media/preprocess \
        -o data/preprocessing/clips.json [--no-encode]

<runs_dir>  holds one directory per recording, each with preprocess/all_data/<frame>/
<seqs.txt>  one recording name per line, in the order the job array used
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import tempfile

FRAME = "rgb_WoArm_WArmObjKpts.png"     # the overlay the pipeline writes for each window frame
# The overlay markers are thin saturated rings a few pixels wide. At 640 px (a 3x downscale)
# they fall below what h264 can hold, and 4:2:0 chroma subsampling then bleeds their colour, so
# the thing the page exists to show is the first thing the encoder throws away. 960 at crf 28
# keeps them distinct and still covers a retina tile.
WIDTH = 960
CRF = 28
FPS = 30
OBJ_LINE = re.compile(r"\[GT masks\] obj\d+ = ([A-Za-z0-9_]+)")

PROVENANCE = {
    "run_date": "2026-10-31",
    "slurm_array": "54855301",
    "humanego_commit": "81888fc",
    "encode": f"{WIDTH} px wide, libx264 crf {CRF}, {FPS} fps",
}


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def objects_of(logs_dir: str, task: int) -> list[str]:
    """What the mask-writing stage said it drew, from the newest job log for this task."""
    best, best_t = None, -1.0
    for p in glob.glob(os.path.join(logs_dir, f"s2_*_{task}.out")):
        try:
            t = os.path.getmtime(p)
        except OSError:
            continue
        if t <= best_t:
            continue
        try:
            txt = open(p, errors="replace").read()
        except OSError:
            continue
        if "[GT masks] obj" in txt:
            best, best_t = txt, t
    return OBJ_LINE.findall(best) if best else []


def encode(frames: list[str], out: str, exe: str) -> None:
    """Concat-demux the frame list so filenames with gaps in their numbering still work."""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        for p in frames:
            fh.write(f"file '{p}'\n")
        listing = fh.name
    try:
        subprocess.run(
            [exe, "-nostdin", "-v", "error", "-r", str(FPS), "-f", "concat", "-safe", "0",
             "-i", listing, "-vf", f"scale={WIDTH}:-2", "-c:v", "libx264", "-crf", str(CRF),
             "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             "-y", out],
            check=True)
    finally:
        os.unlink(listing)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("seqs")
    ap.add_argument("media_dir")
    ap.add_argument("-o", "--out", default="data/preprocessing/clips.json")
    ap.add_argument("--no-encode", action="store_true", help="index clips that already exist")
    ap.add_argument("--rendered", default="", help="ISO date to record; defaults to today")
    a = ap.parse_args()

    seqs = [l.strip() for l in open(a.seqs) if l.strip()]
    logs = os.path.join(a.runs_dir, "logs")
    exe = ffmpeg_exe()

    clips, skipped = [], []
    for task, rec in enumerate(seqs, start=1):
        frames = sorted(glob.glob(os.path.join(a.runs_dir, rec, "preprocess", "all_data", "*", FRAME)))
        if len(frames) < 2:
            skipped.append(rec)
            continue
        out = os.path.join(a.media_dir, f"{rec}.mp4")
        if not a.no_encode and not os.path.exists(out):
            encode(frames, out, exe)
        idx = [int(os.path.basename(os.path.dirname(p))) for p in frames]
        clips.append({
            "rec": rec,
            "participant": rec.split("_")[0],
            "frames": len(frames),
            "start": idx[0],
            "end": idx[-1],
            "objects": objects_of(logs, task),
        })

    if a.rendered:
        rendered = a.rendered
    else:
        from datetime import date
        rendered = date.today().isoformat()

    payload = {
        "schema_version": 1,
        "view": "rectified upright RGB over the manipulation window, with the tracked object "
                "keypoints and hand landmarks drawn on",
        "provenance": {**PROVENANCE, "rendered": rendered},
        "with_ground_truth": len(seqs),
        "without_window": len(skipped),
        "clips": clips,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    print(f"wrote {a.out}: {len(clips)} clips, {len(skipped)} recordings with no window")
    print(f"  frames: {sum(c['frames'] for c in clips):,}")
    print(f"  participants: {len({c['participant'] for c in clips})}, "
          f"objects: {len({o for c in clips for o in c['objects']})}")
    missing = [c["rec"] for c in clips if not c["objects"]]
    if missing:
        print(f"  no object list in the log for {len(missing)}: {missing[:5]}")


if __name__ == "__main__":
    main()
