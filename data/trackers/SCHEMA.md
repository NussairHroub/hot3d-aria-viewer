# HOT3D Aria hand-tracker evaluation — file schemas

`schema_version: hot3d-hand-eval/1`

Everything here is produced by two tracker-agnostic scripts in
`/ibex/project/c2090/hroubna/hot3d_viewer/scripts/`:

| script | role |
|---|---|
| `hot3d_mp_predict.py` | runs MediaPipe Hands over the exported frames, writes one prediction npz per recording |
| `hot3d_hand_score.py` | scores **any** prediction npz in that schema against the GT npz, writes one results JSON |
| `hot3d_display_check.py` | independent check that the scorer's raw→upright rotation agrees with `frames_index.csv` |

Adding WiLoR or HaMeR is therefore: write `pred/<tracker>/<SEQ>.npz` in the schema below,
then run

```
sbatch --export=ALL,TRACKER=wilor,LIMIT=0,OUTNAME=wilor \
       /ibex/project/c2090/hroubna/hot3d_viewer/score.sbatch
```

No scorer change is needed **unless** the tracker's landmark order differs from
MediaPipe's, in which case pass `--lm-map=<21 comma-separated indices>` (see
*Landmark conventions*).

---

## 1. Inputs the scorer assumes

| path | content |
|---|---|
| `/ibex/project/c2090/hroubna/hot3d_eval/gt/<SEQ>.npz` | ground truth, from `hot3d_hand_gt.py` (136 files, 263.4 MB) |
| `/ibex/project/c2090/hroubna/hot3d_eval/gt_seqs.txt` | the 136 annotated Aria recording names, one per line |
| `/ibex/tmp/c2090/hroubna/hot3d_eval/frames/<SEQ>/<frame_index>.jpg` | upright 1408x1408 evaluation frames |
| `/ibex/project/c2090/hroubna/hot3d_eval/frames_index.csv` | which frames were exported, plus GT-presence flags and rotated release boxes |

`frame_index` is the row index on that recording's **RGB annotation timeline** — the same
index the viewer payload arrays and the site's mp4 use. It is the join key everywhere.

---

## 2. Prediction npz — `pred/<tracker>/<SEQ>.npz`

One file per recording. `np.savez_compressed`, `allow_pickle=False`-readable.

Frame-aligned arrays, length `N` = number of exported frames for this recording. **Every**
exported frame appears, including frames with no detection — a miss is a row with
`n_det == 0`, never an absent row.

| key | dtype / shape | meaning |
|---|---|---|
| `frame_index` | int32 (N) | RGB-timeline row index; ascending |
| `timestamp_ns` | int64 (N) | timecode (the annotation clock) |
| `device_timestamp_ns` | int64 (N) | VRS exposure timestamp the frame was decoded from |
| `image_ok` | uint8 (N) | 1 if the JPEG decoded, 0 if it could not be read |
| `n_det` | int8 (N) | number of hands the tracker returned on this frame (0 = recorded miss) |

Detection-aligned arrays, length `M` = `n_det.sum()`, in frame order:

| key | dtype / shape | meaning |
|---|---|---|
| `det_frame_index` | int32 (M) | the frame this detection belongs to |
| `det_label` | uint8 (M) | `0` = tracker said "Left", `1` = "Right" — the tracker's own **image-space** label, not the wearer's hand |
| `det_handedness_score` | float32 (M) | the tracker's confidence in that label |
| `det_conf` | float32 (M) | per-hand detection confidence, or NaN if the tracker does not expose one |
| `det_xy` | float32 (M,21,2) | **image-space landmarks, in upright 1408x1408 pixels** |
| `det_z` | float32 (M,21) | the tracker's relative depth per landmark, tracker-specific units; NaN if none |
| `det_world` | float32 (M,21,3) | root-relative metric landmarks if the tracker produces them, else NaN |

Scalar/metadata strings: `seq`, `tracker`, `tracker_version`, `model_file`, `model_md5`,
`frame_convention`, `landmark_convention`, `det_label_meaning`, `det_conf_note`,
`opts_json`, `runtime_json`, plus `image_wh` (int32, 2).

**Required of any new tracker:** `frame_index`, `n_det`, `det_frame_index`, `det_label`,
`det_handedness_score`, `det_xy`, `image_wh`. The rest are optional; `image_wh` must equal
the GT's `cam_size_wh` or the scorer asserts.

### Coordinate frame — the one thing to get right

`det_xy` is in the **upright** frame: the exported JPEGs are `np.rot90(raw_sensor, k=-1)`,
so a raw sensor pixel `(x, y)` sits at `(H-1-y, x)` with `H = 1408`. This is the frame the
site's mp4s and `frames_index.csv`'s `*_x0..*_y1` boxes live in. It is **not** the frame a
FISHEYE624 projection lands in (that is raw sensor pixels); the scorer does the rotation on
the GT side, so a tracker only ever has to report pixels in the image it was given.

---

## 3. Landmark conventions

**GT ("canonical" / Aria MPS order, 21 landmarks)** — this is what
`hand_tracking_toolkit.metrics.LANDMARKS_TO_EVAL` indexes:

```
0 THUMB_FINGERTIP   5 WRIST                10 INDEX_DISTAL       15 RING_INTERMEDIATE
1 INDEX_FINGERTIP   6 THUMB_INTERMEDIATE   11 MIDDLE_PROXIMAL    16 RING_DISTAL
2 MIDDLE_FINGERTIP  7 THUMB_DISTAL         12 MIDDLE_INTERMEDIATE 17 PINKY_PROXIMAL
3 RING_FINGERTIP    8 INDEX_PROXIMAL       13 MIDDLE_DISTAL      18 PINKY_INTERMEDIATE
4 PINKY_FINGERTIP   9 INDEX_INTERMEDIATE   14 RING_PROXIMAL      19 PINKY_DISTAL
                                                                 20 PALM_CENTER
```

**MediaPipe Hands order (21)**: `0 WRIST, 1-4 thumb CMC/MCP/IP/TIP, 5-8 index
MCP/PIP/DIP/TIP, 9-12 middle, 13-16 ring, 17-20 pinky`.

`hot3d_hand_score.py:MP_FOR_CANON` maps canonical index → prediction index:

```
[4, 8, 12, 16, 20, 0, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
```

20 of 21 landmarks correspond. Canonical `PALM_CENTER` and MediaPipe `THUMB_CMC` have no
counterpart, so **20 landmarks are scored** (`EVAL_IDX = range(20)`), the same 20 the
toolkit's own benchmark evaluates. WiLoR and HaMeR both emit MANO joints; if their output
is already in MediaPipe order the default map applies, otherwise pass `--lm-map=`. The
toolkit ships `MANO_TO_CANONICAL_LANDMARK_MAPPING` for raw MANO joint order — invert it
rather than guessing.

Joint *definitions* still differ between UmeTrack's skinned landmarks and a MANO/MediaPipe
annotation convention, so a few px of systematic offset is baked into every number here.
That is why fingertip-only and wrist-only errors are reported separately: fingertips are
the least convention-dependent points, the wrist the most.

---

## 4. Results JSON — `results/<tracker>.json`

Top level:

| key | meaning |
|---|---|
| `schema_version` | `"hot3d-hand-eval/1"` |
| `tracker`, `created_utc`, `wall_s` | provenance |
| `metric_impl` | `"hand_tracking_toolkit"` when the toolkit's `PCK_curve` / `normalized_AUC` / `compute_mpjpe` were used, else `"local_fallback:<Error>"` |
| `dataset` | roots, recordings requested/scored, `recordings_missing_predictions`, `recordings_with_zero_evaluable_hands`, `recordings_with_no_exported_frames` (`P0015_3a9bb2ae`: the release fails QA on all 1,867 of its RGB frames, so nothing was exported and nothing can be scored; it appears in `per_recording` with zero counts and a `note`) |
| `protocol` | the full gate, projection chain, assignment rule, landmark order and thresholds, in words |
| `handedness` | the left/right mapping experiment (§5) |
| `overall` | the primary metric block |
| `subsets` | `any_landmark_in_view`, `gt_hand_visible`, `gt_hand_not_visible`, `left`, `right` |
| `by_distance_m` | one block per distance bin: `0.00-0.25`, `0.25-0.35`, `0.35-0.45`, `0.45-0.55`, `0.55-0.70`, `0.70-inf` |
| `by_mapping_geometric` | the same block computed with oracle one-to-one assignment (localisation without labelling) |
| `false_positives` | detection counts against frames where GT has no hand annotated or none in view |
| `per_recording` | one row per recording: `seq, participant, frames_scored, gt_hands, matched, detection_rate, mpjpe_2d_px, median_px, frames_with_det, frames` |

### The evaluation gate

A `(frame, hand)` slot is **evaluable** when all of:

1. `mask_qa_pass[frame] == 1` — the release's own QA,
2. `mask_hand_pose_available[frame] == 1` — the release gives that hand a wrist transform
   and its 22 joint angles,
3. the skinned GT landmarks are finite,
4. the headset pose is finite,
5. all 20 scored landmarks project inside the 1408x1408 image (`all_in`).

Condition 5 is the primary subset; `subsets.any_landmark_in_view` relaxes it to ≥1 inside.
`mask_hand_visible` is **not** part of the gate — it is a breakdown axis, because a hand
can be annotated in 3D while occluded or out of RGB view.

### Metric block

Every block has the same shape:

| key | meaning |
|---|---|
| `subset`, `gt_hands`, `matched` | the slots in this block and how many got a detection |
| `detection_rate` | `matched / gt_hands` |
| `mpjpe_2d_px` | mean 2D pixel error over matched pairs × 20 landmarks |
| `mpjpe_2d_px_no_wrist` | same, excluding `WRIST` |
| `mpjpe_2d_px_excl_gross` | same, excluding pairs whose per-hand mean error exceeds 100 px |
| `err_px` | `n, mean, median, p25, p75, p90, p95, max` over all matched landmark errors |
| `err_px_per_frame_mean` | the same stats over per-hand mean errors (one number per matched hand) |
| `fingertip_err_px`, `wrist_err_px` | the same stats restricted to canonical 0-4 / 5 |
| `per_landmark_px` | `{LANDMARK_NAME: {mean, median}}` for all 20 |
| `pck_px_curve`, `pck_auc_px_0_100` | PCK over 101 pixel thresholds 0…100 px; AUC normalised to [0,1] |
| `fingertip_pck_px_curve`, `fingertip_pck_auc_px_0_100` | the same, fingertips only |
| `pck_norm_curve`, `pck_auc_norm_0_0.5` | PCK after dividing each error by that hand's GT landmark bounding-box diagonal; 101 thresholds 0…0.5 |
| `gross_failures_gt100px`, `gross_failure_rate_gt100px` | matched hands whose mean error exceeds 100 px — a detection on the wrong hand or on something that is not a hand |
| `gt_hand_diag_px` | stats of the GT bounding-box diagonal, the normaliser |
| `handedness_score` | stats of the tracker's label confidence on matched hands |

PCK curves are `list[float]` of length 101, percent, x-ticks in
`protocol.px_thresholds` / `protocol.norm_thresholds`.

### Assignment

Primary (`identity` or `swapped`, whichever the handedness experiment picks): a GT hand
takes the highest-`det_handedness_score` detection carrying its mapped label. If both GT
hands end up with the same detection, the cheaper pairing keeps it and the other hand
counts as a miss.

`by_mapping_geometric` instead takes the globally cheapest one-to-one pairing, ignoring
labels. The gap between the two is exactly the cost of the tracker's left/right confusion.

### Distance

`Euclidean distance from the RGB camera centre to the GT WRIST landmark, in metres`, using
the camera-frame point the projection already computed.

---

## 5. The handedness experiment

MediaPipe (and any image-space hand detector) labels hands as they appear in the picture.
In an egocentric view the wearer's own hands enter from the bottom of the frame and the
image-space label need not match the wearer's anatomy. The scorer therefore computes the
whole thing twice:

* `identity` — tracker "Left" → GT left hand
* `swapped` — tracker "Left" → GT right hand

and reports `handedness.comparison` with `gt_hands / matched / detection_rate /
mpjpe_2d_px` for both, plus `geometric` as the label-free reference.
`handedness.chosen` is whichever of the two has the lower `mpjpe_2d_px`, and every block
outside `handedness` and `by_mapping_geometric` uses it.
`label_agreement_with_identity_on_geometric_matches` is the fraction of oracle-matched
hands whose tracker label agreed with the GT side under `identity` — a per-hand labelling
accuracy that does not depend on the choice.

---

## 6. Companion npz — `results/<tracker>_pairs.npz`

The per-pair table behind the JSON, for plots and for the site: `seq` (str), `side` (str),
`frame` (int32), `dist` (float32, m), `diag` (float32, px), `visible` (uint8), `matched`
(uint8), `err` (float32, N×20 — NaN rows where `matched == 0`). One row per evaluable slot
in the primary subset, under the chosen mapping.

---

## 7. Reproduce

```
# predictions (136-task array, batch partition, CPU, ~4 min wall)
sbatch /ibex/project/c2090/hroubna/hot3d_viewer/mp_pred.sbatch

# scoring (single job, ~5 min)
sbatch --export=ALL,TRACKER=mediapipe,LIMIT=0,OUTNAME=mediapipe \
       /ibex/project/c2090/hroubna/hot3d_viewer/score.sbatch

# rotation cross-check
python scripts/hot3d_display_check.py .../frames_index.csv .../gt --pad=8
```

Nothing here needs a GPU and nothing runs on a login node.

---

## 8. Rotation cross-check (why the display frame is trusted)

`hot3d_display_check.py` projects the GT landmarks with the scorer's own code, rotates them
to the upright frame, and tests them against the release's 2D hand boxes **as rotated by a
different script** (`hot3d_eval_frames.py`, whose output is `frames_index.csv`). Result on
every exported frame of all 135 recordings that have frames:

```
TOTAL pad8: 1466850/1466980 = 99.9911%   over 73,349 hand-boxes, 135 recordings
worst recording: P0011_2255f410 99.88%
```

So the scorer's `(x, y) -> (H-1-y, x)` and the exporter's `np.rot90(k=-1)` agree. Any new
tracker inherits this check for free; re-run it if the frame export is ever regenerated.
