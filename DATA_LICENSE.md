# Licences

This repository holds two different things under two different regimes: the site's **code**, and
**material derived from the HOT3D dataset**. The dataset's terms travel with the derived material,
and the repository is laid out so that each licence has its own directory.

## Dataset material — HOT3D (Meta Platforms Technologies, LLC)

The HOT3D Dataset License Agreement (shipped as `license.txt` in every recording, and published at
<https://www.projectaria.com/datasets/hot3d/license/>) splits the release into three data types,
each with its own licence. This repository follows that split exactly:

| Directory | Contents | Licence |
| --- | --- | --- |
| `media/` | RGB video, poster frames, SLAM stills | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| `data/seq/` | Object and headset poses, object 2D boxes, eye gaze, SLAM trajectory, scene points, the five non-hand masks | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| `data/hands/` | UmeTrack wrist poses and joint angles, hand 2D boxes, the two hand masks | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — **non-commercial** |
| `data/objects/` | The 33 object models | CC BY-SA 4.0, **with Meta's rider that the material may not be sold** |
| `data/index.json` | Metadata for all 424 recordings | CC BY-SA 4.0 |

Each of those directories carries its own `LICENSE` file with the attribution and the list of
modifications. The licensor's words for the three types are:

> 1. Sequence captured by Project Aria and Quest devices, and annotations (excluding hand
>    annotations) ("Sequence data") is licensed under CC BY-SA.
> 2. Hand annotations ("Hand data") is licensed under CC BY-NC-SA.
> 3. 3D object models ("Model data") are licensed under a CC BY-SA, with the modification that
>    notwithstanding anything to the contrary, You shall not sell the Licensed Material, or
>    incorporate the Licensed Material into a product to be sold […]

**Why hand data is a separate file.** A single file carrying both regimes cannot be licensed
lawfully: CC BY-SA 4.0 forbids imposing "additional or different terms or conditions" on adapted
material, and the NonCommercial term is exactly such a restriction. So `data/seq/<SEQ>.json` holds
no hand fields, and `data/hands/<SEQ>.hands.json` holds them alone. Any page that displays hand
data — `hands.html`, and the hand layers of `sequence.html`, `scene3d.html` and `quality.html` —
is non-commercial material.

**MANO parameters are not republished here.** The release ships MANO pose coefficients alongside
UmeTrack, but the HOT3D README conditions the use of hand annotations on accepting the
[SMPL-X/MANO licence](https://mano.is.tue.mpg.de/). Only the UmeTrack annotations are served from
this repository; take MANO from the official download.

### Attribution

> HOT3D dataset, © Meta Platforms Technologies, LLC.
> Prithviraj Banerjee et al., *HOT3D: Hand and Object Tracking in 3D from Egocentric Multi-View
> Videos*, CVPR 2025 — <https://arxiv.org/abs/2411.19167>,
> <https://facebookresearch.github.io/hot3d/>
> Dataset and licence: <https://www.projectaria.com/datasets/hot3D/>,
> <https://www.projectaria.com/datasets/hot3d/license/>

Unless separately stated, the licensor offers the licensed material as-is and as-available, and
makes no representations or warranties of any kind concerning it, whether express, implied,
statutory or other. See the licences linked above for the full disclaimer of warranties.

### Changes made to the licensed material

ShareAlike requires that modifications be indicated. This repository:

- re-encodes the RGB stream of 12 of the 198 Aria recordings to 704×704 H.264 at CRF 30, rotated
  90° clockwise to upright, and saves one poster frame and six stills from each SLAM camera;
- resamples the released annotations onto the 30 Hz RGB frame grid by nearest timestamp within
  8 ms, and stores them as base64 float32 arrays;
- rotates the released 2D boxes from sensor pixels into that upright display frame;
- filters the semidense points to `dist_std` ≤ 2 cm and subsamples them evenly to about 60k points
  per recording;
- re-exports the object models with textures resampled to 512 px and geometry decimated to at most
  40k faces;
- splits hand annotations into their own files and omits the MANO parameters.

No original recording, VRS file or unmodified release archive is redistributed here. The full
dataset is obtained from <https://www.projectaria.com/datasets/hot3D/> under its own agreement.

## Site code

`*.html`, `assets/`, `scripts/` and `README.md` are MIT licensed; see [`LICENSE`](LICENSE). That
licence covers the code only — it does not apply to anything under `data/` or `media/`, which
keeps the dataset's terms above.

## Removal requests

If Meta, a HOT3D participant, or anyone with a claim to this material wants it taken down, open an
issue on this repository or write to the address in the GitHub profile of the repository owner, and
it will be removed.
