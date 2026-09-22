#!/usr/bin/env python3
"""Turn one HOT3D Aria recording into a single JSON payload the web viewer can load.

Everything the release ships per recording is folded into one file, indexed by a common
frame timeline (the ground-truth timecode grid):

    headset pose      headset_trajectory.csv          -> T (F,3), Q (F,4 wxyz)
    object poses      dynamic_objects.csv             -> per object T/Q + availability
    hand poses        umetrack / mano jsonl           -> wrist xform, joint angles, MANO betas
    2D boxes          box2d_objects / box2d_hands.csv -> per stream, per frame
    eye gaze          mps/eye_gaze/*.csv              -> yaw/pitch/depth resampled onto the grid
    device trajectory mps/slam/closed_loop*.csv       -> the SLAM path + quality
    scene points      mps/slam/semidense_points.csv.gz-> subsampled xyz
    QA masks          masks/*.csv                     -> per stream, per frame
    calibration       camera_models.json              -> verbatim

Arrays travel as base64 float32/uint8 so the page can decode them straight into typed arrays,
the same convention the HumanEgo viewers use. NaN marks "not annotated on this frame"; the
uint8 masks say why.
"""
import base64
import gzip
import json
import os
import sys

import numpy as np

STREAMS = {"214-1": "rgb", "1201-1": "slam_left", "1201-2": "slam_right"}
MASK_FILES = ["mask_qa_pass", "mask_good_exposure", "mask_hand_visible", "mask_object_visible",
              "mask_hand_pose_available", "mask_headset_pose_available", "mask_object_pose_available"]
# The release licenses hand annotations separately (CC BY-NC-SA) from everything else
# (CC BY-SA), so they are written to their own file and never mixed into the main payload.
HAND_MASKS = ("hand_visible", "hand_pose_available")
# The RGB camera is what the viewer plays, so its boxes travel in the main payload. The two
# monochrome SLAM cameras' boxes are two thirds of that payload's bytes and no page draws them,
# so they are published beside it instead of inside it.
MAIN_BOX_STREAM = "rgb"
LIC_SEQUENCE = ("HOT3D sequence data and non-hand annotations, (c) Meta Platforms Technologies, "
                "LLC, CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/); modified: "
                "resampled onto the RGB frame grid, boxes rotated to the upright frame, points "
                "filtered and subsampled.")
LIC_HANDS = ("HOT3D hand annotations, (c) Meta Platforms Technologies, LLC, CC BY-NC-SA 4.0 "
             "(https://creativecommons.org/licenses/by-nc-sa/4.0/); modified: resampled onto the "
             "RGB frame grid, boxes rotated to the upright frame. MANO parameters are in the "
             "release but are not republished here: their use is gated by the SMPL-X/MANO "
             "license.")
MAX_POINTS = 60000
POINT_DIST_STD_MAX = 0.02  # metres; the MPS-recommended cut for clean semidense points


def b64(a, dtype="<f4"):
    return base64.b64encode(np.ascontiguousarray(a, dtype).tobytes()).decode()


def rnd(x, n=4):
    x = float(x)
    return None if not np.isfinite(x) else round(x, n)


def to_float(col):
    """Annotation CSVs leave a box blank when the object is not seen — those become NaN."""
    return np.asarray([np.nan if s in ("", "None", "nan") else float(s) for s in col],
                      dtype=np.float64)


def read_csv(path, dtype=None):
    """Read a small annotation CSV into a dict of columns (strings kept as object arrays)."""
    with open(path) as f:
        head = f.readline().rstrip("\n").split(",")
        rows = [ln.rstrip("\n").split(",") for ln in f if ln.strip()]
    cols = {}
    for i, name in enumerate(head):
        col = [r[i] for r in rows]
        try:
            cols[name] = np.asarray(col, dtype=dtype or np.float64)
        except ValueError:
            cols[name] = np.asarray(col, dtype=object)
    return cols


def index_of(timeline, ts, tol_ns=8_000_000):
    """Map annotation timestamps onto the frame timeline, dropping anything off-grid."""
    pos = np.searchsorted(timeline, ts)
    pos = np.clip(pos, 1, len(timeline) - 1)
    left, right = timeline[pos - 1], timeline[pos]
    take_left = (ts - left) <= (right - ts)
    idx = np.where(take_left, pos - 1, pos)
    ok = np.abs(timeline[idx] - ts) <= tol_ns
    return idx, ok


def poses_on_timeline(cols, timeline, uid_filter=None):
    """(F,3) translations and (F,4) wxyz quaternions for one object uid, NaN where unannotated."""
    F = len(timeline)
    T = np.full((F, 3), np.nan)
    Q = np.full((F, 4), np.nan)
    sel = slice(None) if uid_filter is None else (cols["object_uid"] == uid_filter)
    ts = to_float(cols["timestamp[ns]"][sel]).astype(np.int64)
    if ts.size == 0:
        return T, Q
    idx, ok = index_of(timeline, ts)
    xyz = np.stack([to_float(cols[f"t_wo_{a}[m]"][sel]) for a in "xyz"], 1)
    quat = np.stack([to_float(cols[f"q_wo_{a}"][sel]) for a in "wxyz"], 1)
    T[idx[ok]] = xyz[ok]
    Q[idx[ok]] = quat[ok]
    return T, Q


def boxes_on_timeline(cols, timeline, stream, key, value, sensor_hw=None):
    """(F,4) xyxy boxes plus (F,) visibility ratio for one stream and one object/hand.

    Boxes are released in raw sensor pixels, but Aria mounts its cameras rotated: the frames are
    turned 90 degrees clockwise for viewing, and the videos on this site are stored that way. The
    boxes are rotated with them here -- verified by drawing a released hand box on a rotated frame
    and seeing it land on the hand -- so a page only ever scales by (video width / display width).
    """
    F = len(timeline)
    box = np.full((F, 4), np.nan)
    vis = np.full(F, np.nan)
    sel = (cols["stream_id"] == stream) & (cols[key] == value)
    if not np.any(sel):
        return box, vis
    ts = cols["timestamp[ns]"][sel].astype(np.int64)
    idx, ok = index_of(timeline, ts)
    xyxy = np.stack([cols[c][sel] for c in
                     ("x_min[pixel]", "y_min[pixel]", "x_max[pixel]", "y_max[pixel]")], 1)
    ok &= np.isfinite(xyxy).all(1)
    if sensor_hw is not None:
        h = sensor_hw[0]
        x0, y0, x1, y1 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
        # (x, y) -> (h - 1 - y, x); corners swap roles, so re-order into xyxy afterwards
        rx0, rx1 = h - 1 - y1, h - 1 - y0
        ry0, ry1 = x0, x1
        xyxy = np.stack([rx0, ry0, rx1, ry1], 1)
    box[idx[ok]] = xyxy[ok]
    vis[idx[ok]] = cols["visibility_ratio[%]"][sel][ok]
    return box, vis


def read_hand_jsonl(path, timeline, key):
    """Wrist transforms plus per-hand parameters from a hand-pose jsonl, on the timeline.

    HOT3D indexes hands as 0 = left, 1 = right, matching box2d_hands.csv.
    """
    F = len(timeline)
    out = {}
    for h in ("0", "1"):
        out[h] = {"wrist_t": np.full((F, 3), np.nan), "wrist_q": np.full((F, 4), np.nan),
                  "params": None, "conf": np.full(F, np.nan)}
    if not os.path.exists(path):
        return out
    stamps, records = [], []
    with open(path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                stamps.append(rec["timestamp_ns"])
                records.append(rec.get("hand_poses", {}))
    if not stamps:
        return out
    idx, ok = index_of(timeline, np.asarray(stamps, dtype=np.int64))
    for h in ("0", "1"):
        n_par = None
        for rec in records:
            if h in rec and key in rec[h]:
                n_par = len(rec[h][key])
                break
        if n_par is not None:
            out[h]["params"] = np.full((F, n_par), np.nan)
    for j, rec in enumerate(records):
        if not ok[j]:
            continue
        i = idx[j]
        for h in ("0", "1"):
            if h not in rec:
                continue
            pose = rec[h]
            w = pose.get("wrist_xform")
            if w:
                out[h]["wrist_t"][i] = w["t_xyz"]
                out[h]["wrist_q"][i] = w["q_wxyz"]
            if out[h]["params"] is not None and key in pose:
                out[h]["params"][i] = pose[key]
            if "hand_confidence" in pose:
                out[h]["conf"][i] = pose["hand_confidence"]
    return out


def gaze_on_timeline(seq_dir, timeline, timecode_to_device):
    """Eye gaze resampled onto the frame timeline, preferring the personalized calibration."""
    for name in ("personalized_eye_gaze.csv", "general_eye_gaze.csv"):
        path = os.path.join(seq_dir, "mps", "eye_gaze", name)
        if os.path.exists(path):
            break
    else:
        return None
    cols = read_csv(path)
    gaze_us = cols["tracking_timestamp_us"].astype(np.int64)
    # the gaze stream is on device time; the frame grid is timecode, so map through the pairing
    device_ns = timecode_to_device(timeline)
    order = np.argsort(gaze_us)
    gaze_us = gaze_us[order]
    pos = np.clip(np.searchsorted(gaze_us, device_ns // 1000), 1, len(gaze_us) - 1)
    near = np.where((device_ns // 1000 - gaze_us[pos - 1]) <= (gaze_us[pos] - device_ns // 1000),
                    pos - 1, pos)
    ok = np.abs(gaze_us[near] - device_ns // 1000) <= 40_000  # 40 ms
    out = {"source": name.replace("_eye_gaze.csv", "")}
    for field, col in (("yaw_l", "left_yaw_rads_cpf"), ("yaw_r", "right_yaw_rads_cpf"),
                       ("pitch", "pitch_rads_cpf"), ("depth", "depth_m")):
        if col not in cols:
            continue
        v = np.full(len(timeline), np.nan)
        # some rows leave a field blank (depth is dropped when the two eyes do not converge)
        v[ok] = to_float(cols[col])[order][near][ok]
        out[field] = v
    return out


def closed_loop_on_timeline(seq_dir, timeline, timecode_to_device):
    """The SLAM device trajectory and its quality score, sampled at the frame timestamps."""
    path = os.path.join(seq_dir, "mps", "slam", "closed_loop_trajectory.csv")
    if not os.path.exists(path):
        return None
    cols = read_csv(path)
    us = cols["tracking_timestamp_us"].astype(np.int64)
    order = np.argsort(us)
    us = us[order]
    device_us = timecode_to_device(timeline) // 1000
    pos = np.clip(np.searchsorted(us, device_us), 1, len(us) - 1)
    near = np.where((device_us - us[pos - 1]) <= (us[pos] - device_us), pos - 1, pos)
    ok = np.abs(us[near] - device_us) <= 20_000
    F = len(timeline)
    T = np.full((F, 3), np.nan)
    Q = np.full((F, 4), np.nan)
    qual = np.full(F, np.nan)
    T[ok] = np.stack([to_float(cols[f"t{a}_world_device"])[order][near][ok] for a in "xyz"], 1)
    Q[ok] = np.stack([to_float(cols[f"q{a}_world_device"])[order][near][ok] for a in "wxyz"], 1)
    if "quality_score" in cols:
        qual[ok] = to_float(cols["quality_score"])[order][near][ok]
    return {"T": T, "Q": Q, "quality": qual,
            "rate_hz": rnd(1e6 * len(us) / max(us[-1] - us[0], 1), 2),
            "rows": int(len(us))}


def semidense_points(seq_dir, max_points=MAX_POINTS):
    """A subsample of the clean semidense points, with the filtering reported alongside."""
    path = os.path.join(seq_dir, "mps", "slam", "semidense_points.csv.gz")
    if not os.path.exists(path):
        return None, None
    xyz, kept, total = [], 0, 0
    with gzip.open(path, "rt") as f:
        head = f.readline().rstrip("\n").split(",")
        ix = [head.index(c) for c in ("px_world", "py_world", "pz_world")]
        istd = head.index("dist_std") if "dist_std" in head else None
        for line in f:
            total += 1
            parts = line.split(",")
            if istd is not None and float(parts[istd]) > POINT_DIST_STD_MAX:
                continue
            kept += 1
            xyz.append([float(parts[i]) for i in ix])
    pts = np.asarray(xyz, dtype=np.float64) if xyz else np.zeros((0, 3))
    stats = {"total": total, "kept": kept, "dist_std_max_m": POINT_DIST_STD_MAX}
    if len(pts) > max_points:  # even stride keeps the room's shape, unlike taking a prefix
        pts = pts[:: int(np.ceil(len(pts) / max_points))]
    stats["shown"] = int(len(pts))
    return pts, stats


def rgb_timeline(seq_dir):
    """The RGB camera's frame timestamps, read from any per-stream annotation file."""
    for rel in ("masks/mask_qa_pass.csv", "masks/mask_good_exposure.csv"):
        path = os.path.join(seq_dir, rel)
        if not os.path.exists(path):
            continue
        cols = read_csv(path, dtype=object)
        ts = cols["timestamp[ns]"][cols["stream_id"] == "214-1"].astype(np.int64)
        if ts.size:
            return np.unique(ts)
    return None


def masks_on_timeline(seq_dir, timeline):
    """Per-stream QA masks as uint8 (1 true, 0 false, 255 unannotated)."""
    out = {}
    mdir = os.path.join(seq_dir, "masks")
    for name in MASK_FILES:
        path = os.path.join(mdir, f"{name}.csv")
        if not os.path.exists(path):
            continue
        cols = read_csv(path, dtype=object)
        ts = cols["timestamp[ns]"].astype(np.int64)
        idx, ok = index_of(timeline, ts)
        val = np.isin(cols["mask"], ("True", "true", "1"))
        per_stream = {}
        for sid, label in STREAMS.items():
            sel = (cols["stream_id"] == sid) & ok
            if not np.any(sel):
                continue
            arr = np.full(len(timeline), 255, dtype=np.uint8)
            arr[idx[sel]] = val[sel].astype(np.uint8)
            per_stream[label] = arr
        if per_stream:
            out[name.replace("mask_", "")] = per_stream
    return out


def main(seq_dir, out_path):
    seq = os.path.basename(seq_dir.rstrip("/"))
    meta = json.load(open(os.path.join(seq_dir, "metadata.json")))
    cams = json.load(open(os.path.join(seq_dir, "camera_models.json")))

    head_cols = read_csv(os.path.join(seq_dir, "headset_trajectory.csv"))
    # The ground truth is annotated at 60 Hz: the RGB and SLAM cameras each run at 30 Hz and their
    # exposures interleave. The viewer plays the RGB stream, so the RGB timestamps are the grid --
    # every other GT row belongs to the SLAM cameras and would double the payload for nothing.
    timeline = rgb_timeline(seq_dir)
    if timeline is None:
        timeline = np.unique(head_cols["timestamp[ns]"].astype(np.int64))
    F = len(timeline)
    t_s = (timeline - timeline[0]) / 1e9

    # timecode -> device time, so the MPS streams (device clock) land on the right frames
    tc = read_csv(os.path.join(seq_dir, "timecode_devicetime_mapping.csv"))
    tc_ns = tc["timecode_ns"].astype(np.int64)
    dev_ns = tc["devicetime_ns"].astype(np.int64)
    order = np.argsort(tc_ns)
    tc_ns, dev_ns = tc_ns[order], dev_ns[order]

    def timecode_to_device(x):
        return np.interp(x.astype(np.float64), tc_ns.astype(np.float64),
                         dev_ns.astype(np.float64)).astype(np.int64)

    sensor_hw = {}
    display_wh = {}
    for cam in cams:
        sid = cam.get("stream_id")
        if sid in STREAMS:
            h, w = int(cam["imageHeight"]), int(cam["imageWidth"])
            sensor_hw[sid] = (h, w)
            display_wh[STREAMS[sid]] = [h, w]  # rotating 90 degrees swaps width and height

    head_T, head_Q = poses_on_timeline(head_cols, timeline)

    obj_cols = read_csv(os.path.join(seq_dir, "dynamic_objects.csv"))
    box_obj = read_csv(os.path.join(seq_dir, "box2d_objects.csv"), dtype=object)
    for c in ("timestamp[ns]", "x_min[pixel]", "x_max[pixel]", "y_min[pixel]", "y_max[pixel]",
              "visibility_ratio[%]"):
        box_obj[c] = to_float(box_obj[c])

    objects = []
    side_boxes = {}          # object uid -> {slam_left|slam_right: box record}
    hand_side_boxes = {}     # hand side  -> {slam_left|slam_right: box record}
    for uid, name, bop in zip(meta["object_uids"], meta["object_names"], meta["object_bop_uids"]):
        T, Q = poses_on_timeline(obj_cols, timeline, uid_filter=float(uid))
        present = np.isfinite(T).all(1)
        entry = {"uid": uid, "name": name, "bop_uid": bop,
                 "T": b64(T), "Q": b64(Q),
                 "frames_with_pose": int(present.sum()),
                 "travel_m": rnd(float(np.nansum(np.linalg.norm(np.diff(T[present], axis=0), axis=1)))
                                 if present.sum() > 1 else 0.0),
                 "boxes": {}}
        for sid, label in STREAMS.items():
            box, vis = boxes_on_timeline(box_obj, timeline, sid, "object_uid", uid,
                                         sensor_hw.get(sid))
            seen = np.isfinite(box).all(1)
            if not seen.any():
                continue
            rec = {"xyxy": b64(box), "vis": b64(vis), "frames_seen": int(seen.sum())}
            if label == MAIN_BOX_STREAM:
                entry["boxes"][label] = rec
            else:
                side_boxes.setdefault(uid, {})[label] = rec
        objects.append(entry)

    box_hand = read_csv(os.path.join(seq_dir, "box2d_hands.csv"), dtype=object)
    for c in ("timestamp[ns]", "x_min[pixel]", "x_max[pixel]", "y_min[pixel]", "y_max[pixel]",
              "visibility_ratio[%]"):
        box_hand[c] = to_float(box_hand[c])

    ume = read_hand_jsonl(os.path.join(seq_dir, "umetrack_hand_pose_trajectory.jsonl"),
                          timeline, "joint_angles")
    mano = read_hand_jsonl(os.path.join(seq_dir, "mano_hand_pose_trajectory.jsonl"),
                           timeline, "pose")

    hands = {}
    for h, side in (("0", "left"), ("1", "right")):
        rec = {"side": side}
        for src, data in (("umetrack", ume), ("mano", mano)):
            d = data[h]
            rec[src] = {"wrist_t": b64(d["wrist_t"]), "wrist_q": b64(d["wrist_q"]),
                        "conf": b64(d["conf"]),
                        "frames": int(np.isfinite(d["wrist_t"]).all(1).sum())}
            if d["params"] is not None:
                rec[src]["params"] = b64(d["params"])
                rec[src]["n_params"] = int(d["params"].shape[1])
        rec["boxes"] = {}
        for sid, label in STREAMS.items():
            box, vis = boxes_on_timeline(box_hand, timeline, sid, "hand_index", str(h),
                                         sensor_hw.get(sid))
            seen = np.isfinite(box).all(1)
            if not seen.any():
                continue
            entry_box = {"xyxy": b64(box), "vis": b64(vis), "frames_seen": int(seen.sum())}
            if label == MAIN_BOX_STREAM:
                rec["boxes"][label] = entry_box
            else:
                hand_side_boxes.setdefault(side, {})[label] = entry_box
        hands[side] = rec

    gaze = gaze_on_timeline(seq_dir, timeline, timecode_to_device)
    slam = closed_loop_on_timeline(seq_dir, timeline, timecode_to_device)
    pts, pt_stats = semidense_points(seq_dir)
    masks = masks_on_timeline(seq_dir, timeline)

    payload = {
        "v": 2, "dataset": "HOT3D", "license": LIC_SEQUENCE, "device": meta.get("headset", "Aria"), "seq": seq,
        "participant": meta.get("participant_id"), "F": F,
        "duration_s": rnd(float(t_s[-1])),
        "fps": rnd(float((F - 1) / t_s[-1])) if t_s[-1] > 0 else None,
        "t": b64(t_s),
        "timecode_ns_first": int(timeline[0]),
        "headset": {"T": b64(head_T), "Q": b64(head_Q),
                    "frames": int(np.isfinite(head_T).all(1).sum())},
        "objects": objects,
        "cameras": cams,
        "streams": STREAMS,
        "display_wh": display_wh,
        "box_frame": "display",  # boxes are in upright (rotated) pixels, not raw sensor pixels
        "have_gt": bool(meta.get("have_hand_object_pose_gt", False)),
    }
    if gaze:
        payload["gaze"] = {k: (v if isinstance(v, str) else b64(v)) for k, v in gaze.items()}
    if slam:
        payload["slam"] = {"T": b64(slam["T"]), "Q": b64(slam["Q"]),
                           "quality": b64(slam["quality"]),
                           "rate_hz": slam["rate_hz"], "rows": slam["rows"]}
    if pts is not None:
        payload["points"] = {"xyz": b64(pts), **pt_stats}
    if masks:
        payload["masks"] = {k: {s: b64(v, "|u1") for s, v in d.items()}
                            for k, d in masks.items() if k not in HAND_MASKS}

    # a compact summary so the index and the tables never have to decode the arrays
    qa = masks.get("qa_pass", {}).get("rgb")
    payload["hands_file"] = f"{seq}.hands.json"
    payload["slam_boxes_file"] = f"{seq}.boxes2d.json"
    payload["summary"] = {
        "frames": F, "duration_s": payload["duration_s"], "fps": payload["fps"],
        "objects": len(objects),
        "object_names": meta["object_names"],
        "qa_pass_rgb": rnd(float(np.mean(qa[qa != 255] == 1)) if qa is not None and np.any(qa != 255) else np.nan),
        "points_kept": pt_stats["kept"] if pt_stats else None,
        "points_shown": pt_stats["shown"] if pt_stats else None,
        "headset_frames": payload["headset"]["frames"],
    }

    hands_payload = {
        "v": 2, "dataset": "HOT3D", "license": LIC_HANDS, "seq": seq, "F": F,
        "participant": meta.get("participant_id"),
        "hands": {side: {k: v for k, v in rec.items() if k != "mano"}
                  for side, rec in hands.items()},
        "masks": {k: {s: b64(v, "|u1") for s, v in d.items()}
                  for k, d in masks.items() if k in HAND_MASKS},
        "slam_boxes_file": f"{seq}.slam_boxes.json",
        "summary": {"frames": F,
                    "hand_frames": {s: hands[s]["umetrack"]["frames"] for s in hands},
                    "mano_frames": {s: hands[s]["mano"]["frames"] for s in hands},
                    "mano_included": False},
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path + ".tmp", "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    os.replace(out_path + ".tmp", out_path)
    data_dir = os.path.dirname(os.path.dirname(out_path))
    boxes_dir = os.path.join(data_dir, "boxes2d")
    os.makedirs(boxes_dir, exist_ok=True)
    boxes_doc = {"v": 1, "dataset": "HOT3D", "license": LIC_SEQUENCE, "seq": seq, "F": F,
                 "note": ("2D boxes in the two monochrome SLAM cameras, in the upright display "
                          "frame. The RGB camera's boxes are in the main payload."),
                 "display_wh": {k: v for k, v in display_wh.items() if k != MAIN_BOX_STREAM},
                 "objects": side_boxes}
    with open(os.path.join(boxes_dir, f"{seq}.boxes2d.json"), "w") as f:
        json.dump(boxes_doc, f, separators=(",", ":"))

    hands_dir = os.path.join(data_dir, "hands")
    os.makedirs(hands_dir, exist_ok=True)
    hand_boxes_doc = {"v": 1, "dataset": "HOT3D", "license": LIC_HANDS, "seq": seq, "F": F,
                      "note": ("2D hand boxes in the two monochrome SLAM cameras, in the upright "
                               "display frame. The RGB camera's boxes are in the hand payload."),
                      "hands": hand_side_boxes}
    with open(os.path.join(hands_dir, f"{seq}.slam_boxes.json"), "w") as f:
        json.dump(hand_boxes_doc, f, separators=(",", ":"))
    hands_path = os.path.join(hands_dir, f"{seq}.hands.json")
    with open(hands_path + ".tmp", "w") as f:
        json.dump(hands_payload, f, separators=(",", ":"))
    os.replace(hands_path + ".tmp", hands_path)
    print(f"[{seq}] F={F} {payload['duration_s']}s objs={len(objects)} "
          f"hands L/R={hands['left']['umetrack']['frames']}/{hands['right']['umetrack']['frames']} "
          f"pts={pt_stats['shown'] if pt_stats else 0} -> {os.path.getsize(out_path)/1e6:.1f} MB "
          f"+ {os.path.getsize(hands_path)/1e6:.1f} MB hands")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
