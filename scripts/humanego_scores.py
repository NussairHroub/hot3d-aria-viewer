#!/usr/bin/env python3
"""Turn the HumanEgo-vs-HOT3D scoring run into the payload humanego.html reads.

The scoring itself happens elsewhere, on the cluster that holds the recordings: HumanEgo's
preprocessing is run over every annotated Aria recording with its reference masks written from
HOT3D's ground-truth object maps, and nearhand/scripts/score_kpts_v2.py scores each triangulated
keypoint against the release. That scorer writes four CSVs; this script reduces them to the
quantiles, curves and per-object rows the page draws, so the browser never parses 3,670 rows of
CSV and the figures on the page are the same numbers the scoring run reported.

Nothing here re-estimates anything. The only arithmetic is aggregation.

Usage:
    python3 scripts/humanego_scores.py <dir with the four CSVs> [-o data/humanego/scores.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import OrderedDict

# a keypoint outside this depth range is a failed reconstruction, not a measurement with a large
# error; the scorer flags it and the page counts it against the within-distance shares
Z_MIN_M, Z_MAX_M = 0.05, 20.0

PROVENANCE = {
    "scored_utc": "2026-10-31T07:17:00Z",
    "slurm_array": "54855301",
    "humanego_commit": "81888fc",
    "scorer": {
        "path": "nearhand/scripts/score_kpts_v2.py",
        "sha256": "d4639e761f9970dff3961cfd59801693c20521e8662980c05975394d1c0baaf4",
    },
    "mask_writer": {
        "path": "nearhand/scripts/write_gt_masks.py",
        "sha256": "3c61d5a6454a1d7d6ea2dab427d7634bc25c457356c0a57607f4539d97314744",
    },
    "versions": {"scipy": "1.17.1", "numpy": "2.4.6", "torch": "2.5.1+cu121"},
}


def rows_of(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def num(v):
    if v is None or v == "":
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def finite(xs):
    return sorted(x for x in xs if isinstance(x, float) and math.isfinite(x))


def quantile(sorted_xs, p):
    """Linear interpolation between order statistics, matching numpy's default."""
    if not sorted_xs:
        return None
    i = (len(sorted_xs) - 1) * p
    lo, hi = math.floor(i), math.ceil(i)
    if lo == hi:
        return sorted_xs[lo]
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (i - lo)


def quantiles(xs, ps=(0.1, 0.25, 0.5, 0.75, 0.9)):
    s = finite(xs)
    return {"n": len(s), **{f"p{int(p * 100)}": quantile(s, p) for p in ps}}


def r(x, d=4):
    return None if x is None or not math.isfinite(x) else round(x, d)


def build(src: str) -> dict:
    kpts = rows_of(os.path.join(src, "keypoints.csv"))
    inst = rows_of(os.path.join(src, "instances.csv"))
    cover = rows_of(os.path.join(src, "coverage.csv"))
    fails = rows_of(os.path.join(src, "failures.csv"))

    diverged = [k for k in kpts if k["diverged"] == "1"]
    live = [k for k in kpts if k["diverged"] == "0"]
    d_surf = [num(k["d_surf_cm"]) for k in live]
    d_ray = [num(k["d_ray_cm"]) for k in live]
    ratio = [num(k["ratio"]) for k in live]

    n_all = len(kpts)
    scored = finite(d_surf)

    def within(t_cm):
        # denominator is every triangulated keypoint: a diverged one can never be within t
        return sum(1 for x in scored if x <= t_cm) / n_all if n_all else None

    summary = {
        "recordings_scored": len({k["rec"] for k in kpts}),
        "object_instances": len(inst),
        "keypoints": n_all,
        "keypoints_diverged": len(diverged),
        "diverged_share": len(diverged) / n_all if n_all else None,
        "instances_with_a_diverged_keypoint": len({(k["rec"], k["obj_key"]) for k in diverged}),
        "scored_share": len(scored) / n_all if n_all else None,
        "d_surf_median_cm": quantile(scored, 0.5),
        "within_cm": {str(t): within(t) for t in (1, 2, 5, 10, 20)},
        "nearest_surface_is_intended": (
            sum(1 for k in live if k["on_intended"] == "1") / len(live) if live else None
        ),
        "nearest_surface_is_intended_2plus": None,
        "landing_pixel_on_intended": None,
        "ratio_median": quantile(finite(ratio), 0.5),
        "dz_median_cm": None,
    }

    amb = [k for k in live if num(k["n_clouds"]) >= 2]
    if amb:
        summary["nearest_surface_is_intended_2plus"] = {
            "value": sum(1 for k in amb if k["on_intended"] == "1") / len(amb),
            "n": len(amb),
        }
    dz = finite([(num(k["z_tri_m"]) - num(k["z_gt_m"])) * 100 for k in live])
    summary["dz_median_cm"] = quantile(dz, 0.5)

    # The curve the page draws: share of ALL keypoints within a distance, on a log grid. It runs
    # out to 10 m rather than stopping at a tidy 1 m so that the curve reaches its own ceiling on
    # the chart instead of being cut off part-way up by the axis.
    grid = [10 ** (-1 + 0.04 * i) for i in range(int(4 / 0.04) + 1)]
    cdf = {"x_cm": [r(x, 4) for x in grid],
           "share": [r(sum(1 for v in scored if v <= x) / n_all, 5) for x in grid]}

    # Spearman between the two error measures: they share almost no machinery, so agreement
    # between them is the check that neither is reading the wrong pixel
    pairs = [(a, b) for a, b in zip(d_ray, d_surf) if math.isfinite(a) and math.isfinite(b)]

    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        k = 0
        while k < len(order):
            j = k
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[k]]:
                j += 1
            avg = (k + j) / 2
            for m in range(k, j + 1):
                out[order[m]] = avg
            k = j + 1
        return out

    spearman = None
    if len(pairs) > 2:
        ra, rb = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
        ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
        cov = sum((a - ma) * (b - mb) for a, b in zip(ra, rb))
        va = sum((a - ma) ** 2 for a in ra)
        vb = sum((b - mb) ** 2 for b in rb)
        spearman = cov / math.sqrt(va * vb) if va > 0 and vb > 0 else None

    # depth-ratio histogram, 0.05 wide, with the tails counted rather than clipped away
    bins = [round(0.5 + 0.05 * i, 2) for i in range(22)]
    rf = finite(ratio)
    hist = [sum(1 for x in rf if lo <= x < lo + 0.05) for lo in bins]
    out_of_range = sum(1 for x in rf if x < 0.5 or x >= 1.6)

    classes = []
    for cls, label in (("compact", "pca₀ ≤ 0.1"), ("middle", "0.1 < pca₀ ≤ 1"),
                       ("spread", "pca₀ > 1")):
        g = [i for i in inst if i["spread_class"] == cls]
        errs = finite([num(i["med_err_cm"]) for i in g])
        classes.append({
            "key": cls,
            "label": label,
            "instances": len(g),
            "all_diverged": sum(1 for i in g if i["med_err_cm"] == ""),
            "median_of_medians_cm": quantile(errs, 0.5),
            "median_extent_m": quantile(finite([num(i["extent_m"]) for i in g]), 0.5),
        })

    inst_med = finite([num(i["med_err_cm"]) for i in inst])
    return {
        "schema_version": 1,
        "created_utc": PROVENANCE["scored_utc"],
        "method": {
            "name": "HumanEgo",
            "stage": "object keypoints: kptsselector → cotracker → camtriangulator",
            "masks": "reference object masks written from HOT3D's ground-truth object maps, "
                     "replacing Grounding DINO, so the method is scored on the object it was asked "
                     "to track rather than on a detector's choice",
            "commit": PROVENANCE["humanego_commit"],
        },
        "dataset": {"name": "HOT3D", "half": "Aria", "release": "v4.0.0"},
        "protocol": {
            "surface_measure": "distance to the nearest point of the intended object's "
                               "ground-truth surface, back-projected from the released depth at "
                               "the window's first frame; no pinhole-to-fisheye pixel table is "
                               "involved",
            "ray_measure": "ground-truth z-depth at the pixel the keypoint projects to in the "
                           "released fisheye camera, compared with the keypoint's own depth",
            "divergence_range_m": [Z_MIN_M, Z_MAX_M],
            "denominator": "within-distance shares are over every triangulated keypoint; "
                           "quantiles are over the keypoints that did not diverge",
        },
        "provenance": PROVENANCE,
        "summary": summary,
        "quantiles": {
            "d_surf_cm": quantiles(d_surf),
            "d_ray_cm": quantiles(d_ray),
            "ratio": quantiles(ratio),
            "instance_median_cm": quantiles(inst_med),
        },
        "cdf": cdf,
        "agreement": {"spearman": r(spearman, 4), "n": len(pairs)},
        "ratio_hist": {"lo": bins, "n": hist, "out_of_range": out_of_range,
                       "out_of_range_share": out_of_range / len(rf) if rf else None},
        "classes": classes,
        "instances": [
            {
                "rec": i["rec"],
                "object": i["object"],
                "cls": i["spread_class"],
                "pca0": r(num(i["pca0"]), 5),
                "extent_m": r(num(i["extent_m"]), 3),
                "err_cm": r(num(i["med_err_cm"]), 3),
                "kept": int(num(i["n_pts"])),
                "diverged": int(num(i["diverged_pts"])),
            }
            for i in inst
        ],
        "scatter": {
            "d_ray_cm": [r(a, 3) for a, _ in pairs],
            "d_surf_cm": [r(b, 3) for _, b in pairs],
        },
        "coverage": [
            {"step": int(num(c["step"])), "stage": c["stage"],
             "recordings": int(num(c["recordings"])), "dropped": int(num(c["dropped"])),
             "reason": c["reason"]}
            for c in cover
        ],
        "failures": [
            {"reason": f["reason"], "recordings": int(num(f["recordings"])),
             "recoverable": f["fixable"]}
            for f in fails
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="directory holding keypoints.csv, instances.csv, coverage.csv, failures.csv")
    ap.add_argument("-o", "--out", default="data/humanego/scores.json")
    a = ap.parse_args()
    payload = build(a.src)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"), allow_nan=False)
    s = payload["summary"]
    print(f"wrote {a.out}  ({os.path.getsize(a.out):,} bytes)")
    print(f"  {s['keypoints']:,} keypoints over {s['object_instances']} object instances, "
          f"{s['recordings_scored']} recordings")
    print(f"  median {s['d_surf_median_cm']:.2f} cm, within 2 cm {100 * s['within_cm']['2']:.1f}%, "
          f"diverged {100 * s['diverged_share']:.1f}%")
    print(f"  spearman(d_ray, d_surf) = {payload['agreement']['spearman']:+.3f}")


if __name__ == "__main__":
    main()
