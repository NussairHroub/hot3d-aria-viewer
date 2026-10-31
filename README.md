# HOT3D Aria Viewer

Interactive views of the Aria half of the [HOT3D](https://facebookresearch.github.io/hot3d/)
dataset (Meta Reality Labs, CVPR 2025): the RGB stream with its released 2D boxes, the
motion-capture ground truth for hands and objects in world coordinates, the headset trajectory and
scene points, eye gaze, the object library, and the per-frame quality masks.

A static site — no build step, no framework. Open `index.html` over any HTTP server.

**Licensing in one line:** the code is MIT; everything under `data/` and `media/` is modified HOT3D
material that keeps the dataset's own licences — CC BY-SA 4.0, and CC BY-NC-SA 4.0 (non-commercial)
for the hand annotations in `data/hands/`, with a no-sale term on the object models. Each of those
directories carries its own `LICENSE`; the details are in [`DATA_LICENSE.md`](DATA_LICENSE.md).

## Pages

| Page | What it shows |
| --- | --- |
| `index.html` | Overview and dataset totals |
| `recordings.html` | All 198 Aria recordings, sortable and filterable |
| `sequence.html` | The video with boxes, masks and a per-frame readout |
| `scene3d.html` | Scene points, headset path, object poses and surface points, and wrists in 3D |
| `hands.html` | UmeTrack and MANO: availability, wrist traces, joint angles |
| `objects.html` | The 33-object library as 3D models, with usage and motion |
| `cameras.html` | The three streams, their calibration and the rig layout |
| `gaze.html` | Yaw, pitch, depth and vergence |
| `quality.html` | The seven masks, per recording and across the release |
| `data.html` | What a recording folder holds and how this site was built from it |
| `trackers.html` | MediaPipe, WiLoR and HaMeR scored against the mocap hands |
| `preprocessing.html` | Each manipulation window, rectified upright, with the tracked points drawn on |

## Layout

```
assets/site.css        the visual system
assets/site.js         HOT3D.* runtime: payload decoding, pose maths, plotting, chrome
data/index.json        metadata for all 424 recordings (198 Aria, 226 Quest 3)
data/seq/<SEQ>.json    per-recording payload, everything except hand annotations
data/hands/<SEQ>.hands.json  hand annotations, licensed separately (non-commercial)
data/objects.json      the object library index
data/objects/*.glb     object models, compacted for the web
data/objects/points.json  400 surface points per model, for the 3D scene's object-point layer
media/<SEQ>_rgb.mp4    704x704 H.264, 30 fps — frame i is row i of every payload array
media/<SEQ>_*.jpg      poster frame and SLAM-camera stills
media/preprocess/<SEQ>.mp4   the manipulation window with the tracked points drawn on, 640 px wide
data/trackers/*.json   hand-tracker scores, one file per tracker (see its SCHEMA.md)
data/preprocessing/clips.json  the clip index: frames, window bounds and objects per recording
scripts/               the extraction pipeline (runs where the dataset is, not in the browser)
```

## Conventions

- Quaternions are `w, x, y, z`; poses map object or device to world; lengths are metres.
- `NaN` in a per-frame array means the frame carries no annotation; in a mask, `255` means the
  same. Neither is a zero.
- 2D boxes are in the **upright display frame**, matching the videos and stills: Aria mounts its
  cameras rotated, so frames are turned 90° clockwise for viewing and the boxes are rotated with
  them (`box_frame: "display"`, `display_wh` per stream).
- The frame timeline is the 30 Hz RGB grid. Ground truth is annotated at 60 Hz because the RGB and
  SLAM cameras interleave; the payload keeps the RGB half.

## Rebuilding the data

The scripts read a downloaded HOT3D release; they do not ship any of it.

```bash
python scripts/hot3d_index.py   <dataset_dir> data/index.json
python scripts/hot3d_extract.py <dataset_dir>/<SEQ> data/seq/<SEQ>.json
python scripts/hot3d_media.py   <dataset_dir>/<SEQ> media --size 704 --crf 30
python scripts/hot3d_objects.py <dataset_dir>/assets/assets data/index.json data/objects data/objects.json
python scripts/hot3d_object_points.py data/objects.json data/objects data/objects/points.json
```

The preprocessing clips come from a pipeline run over the release rather than from the release
alone, so they are built separately, where that run's output lives:

```bash
python scripts/hot3d_preprocess_clips.py <runs_dir> <seqs.txt> media/preprocess -o data/preprocessing/clips.json
```

It encodes each recording's window frames into one clip and writes the index the page reads. It
needs `imageio-ffmpeg` (or ffmpeg on PATH).

`hot3d_extract.py` needs numpy; `hot3d_media.py` needs `projectaria-tools`, Pillow and ffmpeg
(`imageio-ffmpeg` supplies a build with libx264); `hot3d_objects.py` needs trimesh,
`fast-simplification` and Pillow on Python 3.10+. `hot3d_object_points.py` needs only numpy — it
reads the glb containers directly, applying the glTF node hierarchy, and checks itself by
reproducing each model's released `extent_m` (worst disagreement across the 33 models: 0.05 mm).

## Licence

| What | Licence |
| --- | --- |
| `*.html`, `assets/`, `scripts/` | MIT ([`LICENSE`](LICENSE)) |
| `media/`, `data/seq/`, `data/index.json` | CC BY-SA 4.0 |
| `data/hands/` | CC BY-NC-SA 4.0 — non-commercial |
| `data/objects/` | CC BY-SA 4.0, and the models may not be sold |
| `media/preprocess/`, `data/preprocessing/`, `data/trackers/` | CC BY-SA 4.0 — derived from the release's video and annotations |

Hand annotations sit in their own directory because the release licenses them differently, and one
file cannot carry both terms. MANO parameters are not republished here; their use is gated by the
SMPL-X/MANO licence. Attribution, the disclaimer of warranties, the full list of modifications and
a removal-request contact are in [`DATA_LICENSE.md`](DATA_LICENSE.md).

## Provenance and terms

The dataset is the HOT3D release by Meta Reality Labs
([paper](https://arxiv.org/abs/2411.19167), [project page](https://facebookresearch.github.io/hot3d/)),
used under the HOT3D Dataset License Agreement; obtain the dataset itself from
[projectaria.com](https://www.projectaria.com/datasets/hot3D/).
This repository holds derived visualisations of 12 of the 198 Aria recordings — no source
recordings and no VRS files. Object models are the released meshes with textures resampled to
512 px and geometry decimated to at most 40k faces.

Every figure on these pages is computed from the released annotation files at page load. Nothing
is re-estimated from the images.
