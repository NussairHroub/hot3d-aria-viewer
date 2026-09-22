#!/usr/bin/env python3
"""Compact the HOT3D object library for the web and write the object page's index.

The released .glb models carry 2048x2048 textures -- 175 MB for 33 objects, most of it texture
that a thumbnail-sized viewer cannot show. Textures are resampled to 512 and the models are
re-exported; geometry is left untouched so measurements taken from these models still hold.
"""
import glob
import json
import os
import sys
from collections import Counter

import trimesh
from PIL import Image

TEX_MAX = 512
FACE_MAX = 40000  # per object; the heaviest release model is 386k faces, which no web viewer needs


def compact(path, out_path):
    scene = trimesh.load(path, force="scene")
    faces_before = sum(len(g.faces) for g in scene.geometry.values())
    if faces_before > FACE_MAX:
        # decimate every part by the same factor, so the object keeps its proportions
        keep = FACE_MAX / faces_before
        for name in list(scene.geometry):
            g = scene.geometry[name]
            target = max(64, int(len(g.faces) * keep))
            if len(g.faces) > target:
                scene.geometry[name] = g.simplify_quadric_decimation(face_count=target)
    faces = verts = 0
    tex_before = []
    for geom in scene.geometry.values():
        faces += len(geom.faces)
        verts += len(geom.vertices)
        mat = getattr(getattr(geom, "visual", None), "material", None)
        # a model can carry several maps (base colour, emissive, roughness, normal, occlusion)
        # and older exports keep a single image on a SimpleMaterial: shrink whichever exist
        for slot in ("baseColorTexture", "emissiveTexture", "metallicRoughnessTexture",
                     "normalTexture", "occlusionTexture", "image"):
            img = getattr(mat, slot, None)
            if img is None or not hasattr(img, "size"):
                continue
            tex_before.append(img.size)
            if max(img.size) > TEX_MAX:
                scale = TEX_MAX / max(img.size)
                setattr(mat, slot, img.resize(
                    (max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale))),
                    Image.LANCZOS))
    scene.export(out_path)
    bounds = scene.bounds  # metres, the model's own frame
    extents = (bounds[1] - bounds[0]).tolist()
    return {"faces": int(faces), "faces_released": int(faces_before), "vertices": int(verts),
            "textures_before": tex_before, "texture_max": TEX_MAX,
            "extent_m": [round(float(e), 4) for e in extents],
            "diagonal_m": round(float(sum(e * e for e in extents) ** .5), 4),
            "mb_before": round(os.path.getsize(path) / 1e6, 2),
            "mb_after": round(os.path.getsize(out_path) / 1e6, 2)}


def main(assets_dir, index_path, out_dir, out_index):
    os.makedirs(out_dir, exist_ok=True)
    ix = json.load(open(index_path))
    # uid -> name, and how often each object appears, straight from the recordings' metadata
    name_of, recordings, aria_recordings = {}, Counter(), Counter()
    for row in ix["sequences"]:
        for uid, name in zip(row["object_uids"], row["objects"]):
            name_of[uid] = name
            recordings[uid] += 1
            if row["device"] == "Aria":
                aria_recordings[uid] += 1
    bop_of = {uid: bop for row in ix["sequences"]
              for uid, bop in zip(row["object_uids"], row["bop_uids"])}

    out = []
    for path in sorted(glob.glob(os.path.join(assets_dir, "*.glb"))):
        uid = os.path.basename(path)[:-4]
        stats = compact(path, os.path.join(out_dir, f"{uid}.glb"))
        out.append({"uid": uid, "name": name_of.get(uid, f"uid {uid}"),
                    "bop_uid": bop_of.get(uid), "glb": f"{uid}.glb",
                    "recordings": recordings.get(uid, 0),
                    "aria_recordings": aria_recordings.get(uid, 0), **stats})
        print(f"{out[-1]['name']:22s} {stats['mb_before']:6.1f} -> {stats['mb_after']:5.1f} MB "
              f"faces={stats['faces']:7d} diag={stats['diagonal_m']:.3f} m", flush=True)

    out.sort(key=lambda r: -r["aria_recordings"])
    json.dump({"v": 1, "texture_max_px": TEX_MAX, "objects": out},
              open(out_index, "w"), separators=(",", ":"))
    total = sum(r["mb_after"] for r in out)
    print(f"wrote {out_index}: {len(out)} objects, {total:.0f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:5])
