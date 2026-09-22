#!/usr/bin/env python3
"""Render the web viewer's media for one HOT3D Aria recording, straight from the VRS file.

The release ships no preview video for these recordings, so the RGB stream is decoded here and
written as an mp4 on the same frame grid the JSON payload uses -- frame i of the video is row i
of every array in the payload, which is what lets the page scrub both together.

Also writes a poster frame for the sequence browser and a strip of SLAM-camera stills, since the
two 640x480 monochrome cameras are part of what the recording contains.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image
from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId

RGB_STREAM = "214-1"
SLAM_STREAMS = ("1201-1", "1201-2")


def rgb_frames(provider, size):
    """Yield display-oriented RGB frames with their device timestamps.

    Aria stores the RGB sensor rotated 90 degrees clockwise relative to how a person holds the
    scene upright, so every frame is rotated back before it is shown or encoded.
    """
    sid = StreamId(RGB_STREAM)
    n = provider.get_num_data(sid)
    for i in range(n):
        rec = provider.get_image_data_by_index(sid, i)
        img = rec[0].to_numpy_array()
        ts = rec[1].capture_timestamp_ns
        img = np.rot90(img, k=-1)
        im = Image.fromarray(img)
        if size and im.size[0] != size:
            im = im.resize((size, size), Image.BILINEAR)
        yield i, ts, im


def ffmpeg_exe():
    """A build that actually has libx264: Ibex's PATH ffmpeg is a stripped one that rejects -preset."""
    if os.environ.get("FFMPEG"):
        return os.environ["FFMPEG"]
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def encode(frames_dir, out_mp4, fps, crf):
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", f"{fps}",
           "-i", os.path.join(frames_dir, "f%06d.jpg"),
           "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_mp4]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--size", type=int, default=704)
    ap.add_argument("--crf", type=int, default=30)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--poster-only", action="store_true")
    args = ap.parse_args()

    seq = os.path.basename(args.seq_dir.rstrip("/"))
    os.makedirs(args.out_dir, exist_ok=True)
    provider = data_provider.create_vrs_data_provider(os.path.join(args.seq_dir, "recording.vrs"))

    tmp = os.path.join(args.out_dir, f".frames_{seq}")
    if os.path.isdir(tmp):
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
    os.makedirs(tmp, exist_ok=True)
    stamps = []
    poster_path = os.path.join(args.out_dir, f"{seq}_poster.jpg")
    for i, ts, im in rgb_frames(provider, args.size):
        stamps.append(int(ts))
        if i == int(1.0 * args.fps):  # a second in, past the shutter settling
            im.save(poster_path, quality=82)
        if not args.poster_only:
            im.save(os.path.join(tmp, f"f{i:06d}.jpg"), quality=72)
    if not os.path.exists(poster_path) and stamps:
        im.save(poster_path, quality=82)

    out_mp4 = os.path.join(args.out_dir, f"{seq}_rgb.mp4")
    try:
        if not args.poster_only:
            encode(tmp, out_mp4, args.fps, args.crf)
    finally:
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
        os.rmdir(tmp)

    # a few stills from each SLAM camera: evenly spaced, so they show the whole recording
    slam_meta = {}
    for sid_str in SLAM_STREAMS:
        sid = StreamId(sid_str)
        n = provider.get_num_data(sid)
        if n == 0:
            continue
        picks = np.linspace(0, n - 1, 6).astype(int)
        names = []
        for j, k in enumerate(picks):
            rec = provider.get_image_data_by_index(sid, int(k))
            img = np.rot90(rec[0].to_numpy_array(), k=-1)
            name = f"{seq}_{sid_str.replace('-', '_')}_{j}.jpg"
            Image.fromarray(img).save(os.path.join(args.out_dir, name), quality=78)
            names.append(name)
        slam_meta[sid_str] = {"count": int(n), "stills": names}

    info = {"seq": seq, "rgb_frames": len(stamps),
            "video": os.path.basename(out_mp4) if not args.poster_only else None,
            "poster": os.path.basename(poster_path),
            "size_px": args.size, "fps": args.fps, "crf": args.crf,
            "device_timestamps_ns": [stamps[0], stamps[-1]] if stamps else [],
            "slam": slam_meta}
    with open(os.path.join(args.out_dir, f"{seq}_media.json"), "w") as f:
        json.dump(info, f, separators=(",", ":"))
    size_mb = (os.path.getsize(out_mp4) / 1e6) if os.path.exists(out_mp4) else 0.0
    print(f"[{seq}] {len(stamps)} rgb frames -> {size_mb:.1f} MB mp4, "
          f"slam {[ (k, v['count']) for k, v in slam_meta.items() ]}")


if __name__ == "__main__":
    sys.exit(main())
