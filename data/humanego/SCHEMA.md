# `data/humanego/scores.json`

One file, written by `scripts/humanego_scores.py` from the output of the scoring run described
below. `humanego.html` reads it and draws it; no number on that page is computed anywhere else.

Everything here measures an **estimation method** against the HOT3D release. That makes this
folder different from the rest of `data/`, which republishes the release itself.

## What was measured

HumanEgo turns egocentric video into robot training data. One of its preprocessing stages picks
20 points on an object's outline in a reference frame, tracks them with CoTracker across a window,
and triangulates them to 3D. Those points are supposed to lie on the object's surface. HOT3D's
motion-capture annotations say where that surface actually was, so each point can be scored.

Two measures, chosen so that a fault in one would not hide in the other:

- **Surface distance** (`d_surf`) — the ground-truth depth map at the window's first frame is
  back-projected into a world-frame point cloud, one per object. The score is the distance from the
  keypoint to the nearest point of the *intended* object's cloud, in any direction. No
  pinhole-to-fisheye pixel table takes part, so this measure does not depend on the rectification
  at all.
- **Along-ray error** (`d_ray`) — the keypoint is carried into the released fisheye frame by the
  rigid transform between HumanEgo's rectified camera and the released one, projected with the
  FISHEYE624 model, and compared with the ground-truth z-depth stored at that pixel. It is blind to
  sideways error, where the surface measure is not.

`agreement.spearman` is the rank correlation between the two over the keypoints both could score.
Disagreement there is the signal that one of the two is reading the wrong place.

**Divergence.** A keypoint whose depth in the first camera falls outside
`protocol.divergence_range_m` is a failed reconstruction rather than a measurement with a large
error. Quantiles exclude those; within-distance shares count them in the denominator. Both
denominators appear on the page wherever a share is shown.

## Top level

| key | meaning |
| --- | --- |
| `schema_version` | `1` |
| `created_utc` | when the scoring run finished |
| `method` | what was measured: stage, how its masks were obtained, commit |
| `dataset` | which release (`HOT3D`, Aria half, `v4.0.0`) |
| `protocol` | the two measures, the divergence range, and which denominator each figure uses |
| `provenance` | job array, script paths with sha256, library versions |
| `summary` | the headline figures |
| `quantiles` | `d_surf_cm`, `d_ray_cm`, `ratio`, `instance_median_cm`, each `{n, p10, p25, p50, p75, p90}` |
| `cdf` | `{x_cm[], share[]}` — share of **all** keypoints within each distance, on a log grid out to 10 m |
| `agreement` | `{spearman, n}` between the two measures |
| `ratio_hist` | `{lo[], n[], out_of_range, out_of_range_share}` — bins 0.05 wide from 0.5 to 1.6 |
| `classes` | three rows keyed `compact` / `middle` / `spread`, split on `pca0` |
| `instances` | one row per object instance (199) |
| `scatter` | `{d_ray_cm[], d_surf_cm[]}`, index-aligned, keypoints scored by both measures |
| `coverage` | the four steps from the full download to the measured set |
| `failures` | why the recordings that triangulated nothing failed |

### `summary`

| key | meaning |
| --- | --- |
| `recordings_scored`, `object_instances`, `keypoints` | the sizes the figures rest on |
| `keypoints_diverged`, `diverged_share` | failed reconstructions |
| `instances_with_a_diverged_keypoint` | objects touched by at least one |
| `scored_share` | the share with any measurable distance — the ceiling the curve approaches |
| `d_surf_median_cm` | median over the keypoints that did not diverge |
| `within_cm` | `{"1","2","5","10","20"}` → share of **every** triangulated keypoint that close |
| `nearest_surface_is_intended` | the keypoint's nearest ground-truth surface belongs to the object it was asked to track |
| `nearest_surface_is_intended_2plus` | `{value, n}` restricted to frames where at least two objects had ground-truth surface, so a hit is not free |
| `ratio_median`, `dz_median_cm` | depth bias, as a ratio and in centimetres |

### `instances[]`

`rec`, `object`, `cls` (`compact` / `middle` / `spread`), `pca0` (the triangulator's first
eigenvalue for that point set), `extent_m` (the point set's physical bounding-box diagonal),
`err_cm` (that object's median error, `null` when every keypoint diverged), `kept` (points
surviving of 20), `diverged`.

`pca0` is HumanEgo's own number, not something added here. Its log correlates with `extent_m` at
+0.96, and a point set spanning metres cannot be a handheld object — which is what makes it usable
as a reject filter at inference time, with no ground truth needed.

## What this does not cover

- Only recordings where triangulation produced points are scored, so the figures describe the
  easier part of the set and are optimistic for the whole release.
- The reference surface is the *visible* surface at one frame. A keypoint on a face hidden then has
  nothing nearby to be measured against.
- Reference masks come from HOT3D's object maps rather than HumanEgo's usual text-prompted
  detector, so detection error is absent from every figure.
- The 20 keypoints of one object share a mask, a track and a triangulation. They are not
  independent; the per-object medians are the honest unit, and there are 199 of them.

## Regenerating

The scoring itself runs where the recordings live, and writes four CSVs — `keypoints.csv`,
`instances.csv`, `coverage.csv`, `failures.csv`. Then:

```
python3 scripts/humanego_scores.py <dir with those CSVs> -o data/humanego/scores.json
```

`scripts/humanego_scores.py` only aggregates: quantiles, the curve, the histogram and the
per-object rows. It re-estimates nothing, and its `PROVENANCE` block records the run it came from.

## Licence

These are aggregate statistics computed from the released annotations, in the same spirit as the
figures elsewhere on this site. No HOT3D imagery, mesh or per-frame annotation is republished here.
See `DATA_LICENSE.md` at the repository root.
