#!/usr/bin/env python3
"""Scan every downloaded HOT3D recording and write the index the site's browser page reads.

One pass over the small annotation files only -- no VRS decoding -- so the whole release can be
summarised in a couple of minutes: who recorded it, which objects are in it, how long it runs,
how much of it passes QA, and which parts of the release are present on disk.
"""
import json
import os
import sys
from collections import Counter

STREAM_RGB = "214-1"


def count_rgb_frames_and_qa(seq_dir):
    """RGB frame count and the share of RGB frames that pass QA, from the mask files."""
    path = os.path.join(seq_dir, "masks", "mask_qa_pass.csv")
    if not os.path.exists(path):
        return None, None
    n = passed = 0
    with open(path) as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 3 or parts[1] != STREAM_RGB:
                continue
            n += 1
            passed += parts[2] in ("True", "true", "1")
    return n, (passed / n if n else None)


def duration_s(seq_dir):
    path = os.path.join(seq_dir, "headset_trajectory.csv")
    first = last = None
    with open(path) as f:
        f.readline()
        for line in f:
            parts = line.split(",")
            if len(parts) < 2:
                continue
            t = int(parts[1])
            first = t if first is None else min(first, t)
            last = t if last is None else max(last, t)
    return round((last - first) / 1e9, 2) if first is not None else None


def main(root, out_path):
    seqs = sorted(d for d in os.listdir(root) if d.startswith("P0") and
                  os.path.isdir(os.path.join(root, d)))
    rows, objects_seen, participants = [], Counter(), Counter()
    for seq in seqs:
        d = os.path.join(root, seq)
        meta_path = os.path.join(d, "metadata.json")
        if not os.path.exists(meta_path):
            continue
        meta = json.load(open(meta_path))
        device = meta.get("headset", "?")
        frames, qa = count_rgb_frames_and_qa(d)
        has = {
            "vrs": os.path.exists(os.path.join(d, "recording.vrs")),
            "mps_slam": os.path.exists(os.path.join(d, "mps", "slam", "closed_loop_trajectory.csv")),
            "mps_points": os.path.exists(os.path.join(d, "mps", "slam", "semidense_points.csv.gz")),
            "eye_gaze": os.path.exists(os.path.join(d, "mps", "eye_gaze", "general_eye_gaze.csv")),
            "umetrack": os.path.exists(os.path.join(d, "umetrack_hand_pose_trajectory.jsonl")),
            "mano": os.path.exists(os.path.join(d, "mano_hand_pose_trajectory.jsonl")),
        }
        row = {
            "seq": seq, "device": device, "participant": meta.get("participant_id"),
            "objects": meta.get("object_names", []), "object_uids": meta.get("object_uids", []),
            "bop_uids": meta.get("object_bop_uids", []),
            "have_gt": bool(meta.get("have_hand_object_pose_gt", False)),
            "version": meta.get("version"),
            "rgb_frames": frames, "qa_pass_rgb": round(qa, 4) if qa is not None else None,
            "duration_s": duration_s(d),
            "vrs_gb": round(os.path.getsize(os.path.join(d, "recording.vrs")) / 1e9, 3)
            if has["vrs"] else None,
            "has": has,
        }
        rows.append(row)
        if device == "Aria":
            participants[row["participant"]] += 1
            objects_seen.update(row["objects"])
        print(f"{seq} {device} {row['duration_s']}s {frames} rgb frames "
              f"qa={row['qa_pass_rgb']} objs={len(row['objects'])}", flush=True)

    aria = [r for r in rows if r["device"] == "Aria"]
    out = {
        "v": 1, "root": root, "sequences": rows,
        "totals": {
            "recordings": len(rows),
            "aria": len(aria),
            "quest": sum(1 for r in rows if r["device"] != "Aria"),
            "aria_minutes": round(sum(r["duration_s"] or 0 for r in aria) / 60, 1),
            "aria_rgb_frames": sum(r["rgb_frames"] or 0 for r in aria),
            "participants": len(participants),
            "objects": len(objects_seen),
        },
        "object_counts": dict(objects_seen.most_common()),
        "participant_counts": dict(sorted(participants.items())),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, "w"), separators=(",", ":"))
    print("wrote", out_path, json.dumps(out["totals"]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
