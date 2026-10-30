#!/usr/bin/env python3
"""Sample surface points from each object model, for the 3D scene's object-point layer.

The scene page can place an object's pose, but a wireframe box and an axis triad say nothing
about the object's SHAPE. The released meshes do, and they are already in data/objects — but
they total 29.7 MB, far too much for a page that also fetches a 60k-point scene cloud. So
the surface is sampled here, once, into a few hundred points per object.

Points are AREA-WEIGHTED samples over the triangles, not vertices: a decimated mesh puts its
vertices where curvature is, so a vertex sample clusters on detail and leaves flat faces bare,
which reads as holes once the cloud is posed in the scene.

Coordinates are the model's own frame -- the same frame the released 6-DoF poses act on, which
is what makes `point_world = q * p + t` with the payload's own objects[].T/.Q correct. The glTF
node hierarchy IS applied while reading, because a node transform left out would displace every
point of that mesh. The check that this came out right is printed per object: the sampled
bounding box must reproduce the released `extent_m` in data/objects.json.

No external mesh library: this reads the glb containers directly (numpy only).
"""
import base64
import json
import os
import struct
import sys

import numpy as np

COMP = {5120: ("i1", 1), 5121: ("u1", 1), 5122: ("<i2", 2), 5123: ("<u2", 2),
        5125: ("<u4", 4), 5126: ("<f4", 4)}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def read_glb(path):
    """(gltf json, binary chunk) from a .glb container."""
    with open(path, "rb") as f:
        buf = f.read()
    magic, _ver, _len = struct.unpack_from("<III", buf, 0)
    if magic != 0x46546C67:
        raise ValueError(f"{path}: not a glb")
    off, js, bin_ = 12, None, b""
    while off + 8 <= len(buf):
        clen, ctype = struct.unpack_from("<II", buf, off)
        data = buf[off + 8: off + 8 + clen]
        if ctype == 0x4E4F534A:
            js = json.loads(data.decode("utf-8"))
        elif ctype == 0x004E4942:
            bin_ = data
        off += 8 + clen + ((4 - clen % 4) % 4 if clen % 4 else 0)
    if js is None:
        raise ValueError(f"{path}: no JSON chunk")
    return js, bin_


def accessor(js, bin_, idx):
    acc = js["accessors"][idx]
    dt, sz = COMP[acc["componentType"]]
    n = NCOMP[acc["type"]]
    if "bufferView" not in acc:
        return np.zeros((acc["count"], n), dtype=np.dtype(dt))
    bv = js["bufferViews"][acc["bufferView"]]
    base = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride") or n * sz
    if stride == n * sz:
        a = np.frombuffer(bin_, dtype=np.dtype(dt), count=acc["count"] * n, offset=base)
        return a.reshape(acc["count"], n)
    out = np.empty((acc["count"], n), dtype=np.dtype(dt))
    for k in range(acc["count"]):
        out[k] = np.frombuffer(bin_, dtype=np.dtype(dt), count=n, offset=base + k * stride)
    return out


def node_matrix(node):
    if "matrix" in node:                       # glTF stores column-major
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T
    M = np.eye(4)
    if "scale" in node:
        M[:3, :3] = M[:3, :3] @ np.diag(node["scale"])
    if "rotation" in node:
        x, y, z, w = node["rotation"]          # glTF quaternion order is x,y,z,w
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        M[:3, :3] = R @ M[:3, :3]
    if "translation" in node:
        M[:3, 3] = node["translation"]
    return M


def triangles(path):
    """All triangles of the file in the MODEL frame, with the node hierarchy applied."""
    js, bin_ = read_glb(path)
    tris = []
    scene = js.get("scenes", [{}])[js.get("scene", 0)]
    roots = scene.get("nodes", list(range(len(js.get("nodes", [])))))

    def walk(ni, parent):
        node = js["nodes"][ni]
        M = parent @ node_matrix(node)
        if "mesh" in node:
            for prim in js["meshes"][node["mesh"]].get("primitives", []):
                if prim.get("mode", 4) != 4:          # triangles only
                    continue
                if "POSITION" not in prim.get("attributes", {}):
                    continue
                P = accessor(js, bin_, prim["attributes"]["POSITION"]).astype(np.float64)
                P = (M[:3, :3] @ P.T).T + M[:3, 3]
                if "indices" in prim:
                    I = accessor(js, bin_, prim["indices"]).reshape(-1).astype(np.int64)
                else:
                    I = np.arange(len(P), dtype=np.int64)
                I = I[: (len(I) // 3) * 3].reshape(-1, 3)
                tris.append(P[I])
        for c in node.get("children", []):
            walk(c, M)

    for r in roots:
        walk(r, np.eye(4))
    if not tris:
        raise ValueError(f"{path}: no triangles")
    return np.concatenate(tris, 0)


def sample(tris, n, seed):
    """Area-weighted surface samples, so flat faces are covered as well as detailed ones."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    tot = area.sum()
    if tot <= 0:
        raise ValueError("degenerate mesh")
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(tris), size=n, p=area / tot)
    u = rng.random((n, 1))
    v = rng.random((n, 1))
    over = (u + v) > 1.0
    u[over] = 1.0 - u[over]
    v[over] = 1.0 - v[over]
    A, B, C = a[pick], b[pick], c[pick]
    return A + u * (B - A) + v * (C - A)


def main(objects_json="data/objects.json", glb_dir="data/objects",
         out="data/objects/points.json", n=400):
    n = int(n)
    lib = json.load(open(objects_json))
    rows, worst = {}, (0.0, None)
    for o in lib["objects"]:
        path = os.path.join(glb_dir, o["glb"])
        if not os.path.exists(path):
            print(f"  {o['name']:24s} MISSING {o['glb']}")
            continue
        tris = triangles(path)
        # the seed is the uid, so re-running reproduces the same cloud byte for byte
        pts = sample(tris, n, seed=int(o["uid"]) % (2 ** 32))
        ext = np.asarray(o["extent_m"], dtype=np.float64)
        got = pts.max(0) - pts.min(0)
        # a sample never quite reaches the extreme vertices; compare against the TRIANGLE
        # bounds, which must match the release, and report how much the sample shrinks it
        tb = tris.reshape(-1, 3)
        full = tb.max(0) - tb.min(0)
        err = float(np.max(np.abs(full - ext)))
        shrink = float(np.max(1.0 - got / np.maximum(full, 1e-9)))
        if err > worst[0]:
            worst = (err, o["name"])
        rows[o["uid"]] = {
            "name": o["name"], "n": int(n),
            "xyz": base64.b64encode(pts.astype("<f4").tobytes()).decode("ascii"),
            "extent_m": [round(float(x), 5) for x in full],
        }
        print(f"  {o['name']:24s} {len(tris):6d} tris  bbox err {err*1000:5.2f} mm  "
              f"sample shrink {shrink*100:4.1f}%")
    doc = {"v": 1, "n_per_object": n,
           "frame": ("the model's own frame, the same frame the released 6-DoF object poses "
                     "act on: point_world = quaternion(objects[].Q) * p + objects[].T"),
           "method": ("area-weighted samples over the glb triangles, with the glTF node "
                      "hierarchy applied; seeded per object uid so the cloud is reproducible"),
           "licence": ("derived from the HOT3D object models: CC BY-SA 4.0 with the licensor's "
                       "no-sale modification. See the LICENSE in this directory."),
           "objects": rows}
    json.dump(doc, open(out, "w"), separators=(",", ":"))
    print(f"\nwrote {out}: {len(rows)} objects x {n} points, "
          f"{os.path.getsize(out)/1e3:.0f} kB")
    print(f"worst bounding-box disagreement with the released extent_m: "
          f"{worst[0]*1000:.2f} mm ({worst[1]})")


if __name__ == "__main__":
    main(*sys.argv[1:])
