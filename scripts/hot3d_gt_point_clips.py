#!/usr/bin/env python3
"""Index the ground-truth point clips for `gt_points.html`.

The clips themselves are rendered on the cluster, where the ground-truth object maps live: for
each frame, the object's pixels are resampled into the rectified upright frame, the longest
outline is taken, and twenty points are spaced evenly along it by arc length. This script does
not render; it writes the index the page reads.

Each clip is a WHOLE recording, every third frame played at 30 fps, so about two minutes of
footage runs in about forty seconds. A full recording at full rate would be 8.7x the frames of
the manipulation window alone, which no static site can carry for 119 recordings.

Which objects each recording outlines still comes from the preprocessing index, so the two
galleries pair tile for tile. The figure added per object is how often the release itself says
that object is visible ACROSS THE WHOLE RECORDING -- much lower than over the window it was
chosen in, and the honest answer to "why does this clip spend so long with no points on it".

The clips are too large for the site's own 1 GB budget, so both tiers live on Hugging Face and
the page streams them from there. This script records which files exist and how big they are, so
a caption can offer the native file without the reader guessing what they are about to open.

Usage:
    python3 scripts/hot3d_gt_point_clips.py <gt_archives_dir> data/preprocessing/clips.json \
        <light_dir> --native-dir <native_dir> --base-url <url> -o data/gtpoints/clips.json
"""

from __future__ import annotations

import argparse
import io
import json
import os
import zipfile

import numpy as np

PROVENANCE = {
    "source": "HOT3D released object poses and meshes, rendered to per-frame object-index maps",
    "points_per_object": 20,
    "placement": "evenly spaced by arc length along the object's longest outline, per frame",
    "min_pixels": 30,
    "extent": "the whole recording, at real-time speed",
    "stride": 1,
    "encode": "light 960 px crf 28; native 1920x1280 crf 20; both 30 fps",
    "hosting": "Hugging Face dataset Nussair1/hot3d-aria-gt-points",
}
STRIDE = 1
FPS = 30


def visible_share(archive: str, rec: str, name: str, lo=None, hi=None):
    """The share of frames in which the release says this object is visible.

    With no bounds this covers the whole recording, which is what the clip now shows.
    """
    p = os.path.join(archive, f"{rec}.zip")
    if not os.path.exists(p):
        return None
    try:
        z = zipfile.ZipFile(p)
        objs = np.load(io.BytesIO(z.read("objects.npy")))
        names = list(json.load(z.open("meta.json"))["object_names"])
    except Exception:
        return None
    if name not in names:
        return None
    oi = names.index(name)
    sel = objs[objs["obj_index"] == oi]
    if lo is not None:
        sel = sel[(sel["i"] >= lo) & (sel["i"] <= hi)]
    if sel.size == 0:
        return None
    return float(np.mean(sel["visible_px"] > 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("archives", help="directory of <rec>.zip ground-truth maps")
    ap.add_argument("preprocessing_index", help="data/preprocessing/clips.json")
    ap.add_argument("--site-index", default="data/index.json",
                    help="for each recording's true length")
    ap.add_argument("media_dir", help="the light-tier clips")
    ap.add_argument("--native-dir", default="", help="the native-tier clips")
    ap.add_argument("--base-url", default="", help="where both tiers are served from")
    ap.add_argument("-o", "--out", default="data/gtpoints/clips.json")
    ap.add_argument("--rendered", default="")
    a = ap.parse_args()

    base = json.load(open(a.preprocessing_index))
    length = {}
    if os.path.exists(a.site_index):
        for r in json.load(open(a.site_index))["sequences"]:
            length[r["seq"]] = (r["rgb_frames"], r["duration_s"])
    clips, missing = [], []
    for c in base["clips"]:
        light = os.path.join(a.media_dir, f"{c['rec']}.mp4")
        if not os.path.exists(light):
            missing.append(c["rec"])
            continue
        native = os.path.join(a.native_dir, f"{c['rec']}.mp4") if a.native_dir else ""
        tracked = []
        for t in c.get("tracked", []):
            share = visible_share(a.archives, c["rec"], t["name"])
            win = visible_share(a.archives, c["rec"], t["name"], c["start"], c["end"])
            tracked.append({
                "key": t["key"],
                "name": t["name"],
                # how often the release says it is visible; the clip can only draw it then
                "visible_share": None if share is None else round(share, 4),
                "visible_in_window": None if win is None else round(win, 4),
            })
        rec_frames, dur = length.get(c["rec"], (None, None))
        clips.append({
            "rec": c["rec"],
            "participant": c["participant"],
            "recording_frames": rec_frames,
            "duration_s": dur,
            "clip_seconds": None if rec_frames is None else round(rec_frames / STRIDE / FPS, 1),
            "light_mb": round(os.path.getsize(light) / 1048576, 1),
            "native_mb": (round(os.path.getsize(native) / 1048576, 1)
                          if native and os.path.exists(native) else None),
            "window_frames": c["frames"],
            "window_start": c["start"],
            "window_end": c["end"],
            "objects": [t["name"] for t in tracked],
            "tracked": tracked,
        })

    if a.rendered:
        rendered = a.rendered
    else:
        from datetime import date
        rendered = date.today().isoformat()

    payload = {
        "schema_version": 1,
        "view": "the whole recording in the rectified upright frame, at real-time speed, with "
                "twenty ground-truth points per object placed on its outline in every frame",
        "base_url": a.base_url.rstrip("/"),
        "provenance": {**PROVENANCE, "rendered": rendered},
        "with_ground_truth": base.get("with_ground_truth"),
        "without_window": base.get("without_window"),
        "clips": clips,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))

    vis = [t["visible_share"] for c in clips for t in c["tracked"] if t["visible_share"] is not None]
    secs = [c["clip_seconds"] for c in clips if c["clip_seconds"]]
    print(f"wrote {a.out}: {len(clips)} clips, {sum(len(c['tracked']) for c in clips)} objects")
    if missing:
        print(f"  no rendered clip for {len(missing)}: {missing[:5]}")
    if vis:
        print(f"  object visible over the whole recording: median {100 * float(np.median(vis)):.0f}%, "
              f"worst {100 * min(vis):.0f}%")
    if secs:
        print(f"  clip length: median {float(np.median(secs)):.0f} s, "
              f"longest {max(secs):.0f} s, total {sum(secs)/60:.0f} min")
    lt = sum(c["light_mb"] for c in clips)
    nt = sum(c["native_mb"] for c in clips if c["native_mb"])
    n_native = sum(1 for c in clips if c["native_mb"])
    print(f"  light tier: {lt/1024:.2f} GB over {len(clips)} clips")
    print(f"  native tier: {nt/1024:.2f} GB over {n_native} clips"
          + ("" if n_native == len(clips) else "  (INCOMPLETE)"))


if __name__ == "__main__":
    main()
