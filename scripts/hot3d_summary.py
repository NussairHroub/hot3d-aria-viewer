#!/usr/bin/env python3
"""Precompute the cross-recording figures so the overview pages need one small file, not twelve big ones.

The object, hand, gaze and quality pages all show the same shape of thing: one row per featured
recording, aggregated over its frames. Computing that in the browser meant fetching every payload
(46-66 MB per page). The same numbers are computed here once, from the published payload pair, and
written to data/summary.json (a few hundred KB), leaving the full payload to be fetched only for
the recording a reader actually selects.

Every figure here is defined exactly as the pages defined it, including the NaN handling: a frame
with no annotation is skipped, never counted as zero, and path lengths do not bridge gaps.
"""
import base64
import glob
import json
import os
import sys

import numpy as np

STREAMS = ("rgb", "slam_left", "slam_right")
MASKS = ("qa_pass", "good_exposure", "object_visible", "headset_pose_available",
         "object_pose_available", "hand_visible", "hand_pose_available")


def dec(b64, cols):
    a = np.frombuffer(base64.b64decode(b64), "<f4").astype(np.float64)
    return a.reshape(-1, cols) if cols > 1 else a


def u8(b64):
    return np.frombuffer(base64.b64decode(b64), np.uint8)


def rnd(x, n=4):
    x = float(x)
    return None if not np.isfinite(x) else round(x, n)


def nanmed(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else np.nan


def path_length(T):
    """Metres travelled over annotated samples; a gap is skipped, not crossed in a straight line."""
    ok = np.isfinite(T).all(1)
    step = np.linalg.norm(np.diff(T, axis=0), axis=1)
    both = ok[:-1] & ok[1:]
    return float(step[both].sum()) if both.any() else 0.0


def speed(T, t):
    """Per-frame speed in m/s, NaN wherever either end of the step is unannotated."""
    v = np.full(len(T), np.nan)
    ok = np.isfinite(T).all(1)
    dt = np.diff(t)
    step = np.linalg.norm(np.diff(T, axis=0), axis=1)
    good = ok[:-1] & ok[1:] & (dt > 0)
    v[1:][good] = step[good] / dt[good]
    return v


def mask_rates(block):
    """Pass rate per stream, counted over annotated frames only (255 is not a failure)."""
    out = {}
    for stream, b in (block or {}).items():
        m = u8(b)
        seen = m != 255
        out[stream] = {"annotated": int(seen.sum()), "frames": int(m.size),
                       "rate": rnd(float((m[seen] == 1).mean()) if seen.any() else np.nan)}
    return out


def summarise(seq_path, hands_path):
    d = json.load(open(seq_path))
    h = json.load(open(hands_path)) if os.path.exists(hands_path) else None
    F = d["F"]
    t = dec(d["t"], 1)
    head_T = dec(d["headset"]["T"], 3)
    head_v = speed(head_T, t)

    objects = []
    for o in d["objects"]:
        T = dec(o["T"], 3)
        posed = np.isfinite(T).all(1)
        dist = np.full(F, np.nan)
        both = posed & np.isfinite(head_T).all(1)
        dist[both] = np.linalg.norm(T[both] - head_T[both], axis=1)
        box = o["boxes"].get("rgb")
        vis = dec(box["vis"], 1) if box else np.full(F, np.nan)
        xyxy = dec(box["xyxy"], 4) if box else np.full((F, 4), np.nan)
        boxed = np.isfinite(xyxy).all(1)
        objects.append({
            "uid": o["uid"], "name": o["name"], "bop_uid": o["bop_uid"],
            "frames_with_pose": int(posed.sum()),
            "travel_m": rnd(path_length(T)),
            "travel_m_released": o.get("travel_m"),   # bridges gaps; kept only for comparison
            "frames_boxed_rgb": int(boxed.sum()),
            "median_vis_rgb": rnd(nanmed(vis)),
            "median_dist_headset_m": rnd(nanmed(dist)),
            "max_speed_m_s": rnd(np.nanmax(speed(T, t)) if posed.sum() > 1 else np.nan),
        })

    hands = {}
    if h:
        for side, rec in h["hands"].items():
            ume = rec["umetrack"]
            W = dec(ume["wrist_t"], 3)
            conf = dec(ume["conf"], 1)
            box = rec["boxes"].get("rgb")
            xyxy = dec(box["xyxy"], 4) if box else np.full((F, 4), np.nan)
            hands[side] = {
                "pose_frames": int(np.isfinite(W).all(1).sum()),
                "box_frames_rgb": int(np.isfinite(xyxy).all(1).sum()),
                "box_without_pose": int((np.isfinite(xyxy).all(1) & ~np.isfinite(W).all(1)).sum()),
                "mano_frames": h["summary"]["mano_frames"].get(side),
                "median_conf": rnd(nanmed(conf)),
                "median_height_m": rnd(nanmed(W[:, 2])),
                "median_speed_m_s": rnd(nanmed(speed(W, t))),
                "travel_m": rnd(path_length(W)),
            }
        L = dec(h["hands"]["left"]["umetrack"]["wrist_t"], 3)
        R = dec(h["hands"]["right"]["umetrack"]["wrist_t"], 3)
        both = np.isfinite(L).all(1) & np.isfinite(R).all(1)
        sep = np.full(F, np.nan)
        sep[both] = np.linalg.norm(L[both] - R[both], axis=1)
        hands["separation"] = {"median_m": rnd(nanmed(sep)), "frames": int(both.sum())}

    gaze = None
    if "gaze" in d:
        g = d["gaze"]
        yl, yr = dec(g["yaw_l"], 1), dec(g["yaw_r"], 1)
        pitch, depth = dec(g["pitch"], 1), dec(g["depth"], 1)
        verg = yl - yr
        gaze = {
            "source": g.get("source"),
            "frames_with_gaze": int(np.isfinite(pitch).sum()),
            "median_yaw_l": rnd(nanmed(yl)), "median_yaw_r": rnd(nanmed(yr)),
            "median_pitch": rnd(nanmed(pitch)), "median_depth_m": rnd(nanmed(depth)),
            "median_vergence": rnd(nanmed(verg)),
            "depth_p10": rnd(np.nanpercentile(depth, 10) if np.isfinite(depth).any() else np.nan),
            "depth_p90": rnd(np.nanpercentile(depth, 90) if np.isfinite(depth).any() else np.nan),
        }

    masks = {}
    for name in MASKS:
        block = (d.get("masks") or {}).get(name) or ((h or {}).get("masks") or {}).get(name)
        if block:
            masks[name] = mask_rates(block)

    return {
        "seq": d["seq"], "participant": d["participant"], "F": F,
        "duration_s": d["duration_s"], "fps": d["fps"],
        "qa_pass_rgb": d["summary"].get("qa_pass_rgb"),
        "objects": objects,
        "headset": {"travel_m": rnd(path_length(head_T)),
                    "median_speed_m_s": rnd(nanmed(head_v)),
                    "frames": d["headset"]["frames"]},
        "points": {"kept": d["points"]["kept"], "shown": d["points"]["shown"]} if "points" in d else None,
        "hands": hands or None,
        "gaze": gaze,
        "masks": masks,
        "bytes": {"seq": os.path.getsize(seq_path),
                  "hands": os.path.getsize(hands_path) if os.path.exists(hands_path) else 0},
    }


def main(root="data", out="data/summary.json"):
    rows = []
    for seq_path in sorted(glob.glob(os.path.join(root, "seq", "*.json"))):
        seq = os.path.basename(seq_path)[:-5]
        hands_path = os.path.join(root, "hands", f"{seq}.hands.json")
        rows.append(summarise(seq_path, hands_path))
        print(f"{seq}: {len(rows[-1]['objects'])} objects, "
              f"hands {'yes' if rows[-1]['hands'] else 'no'}, "
              f"{len(rows[-1]['masks'])} masks", flush=True)
    doc = {"v": 1, "note": ("Per-recording aggregates for the twelve featured Aria recordings, "
                            "computed from the published payloads by scripts/hot3d_summary.py. "
                            "Hand figures derive from CC BY-NC-SA material; the rest is CC BY-SA."),
           "recordings": rows}
    json.dump(doc, open(out, "w"), separators=(",", ":"))
    print(f"wrote {out}: {len(rows)} recordings, {os.path.getsize(out)/1e3:.0f} KB")


if __name__ == "__main__":
    main(*sys.argv[1:])
