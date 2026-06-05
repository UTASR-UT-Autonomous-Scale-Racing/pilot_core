#!/usr/bin/env python3
"""Make a copy of the stock AutoDRIVE sim with feature-rich textures, for visual SLAM.

The stock build gives ORB-SLAM3 almost nothing to track (docs/autodrive.md): the track
walls are the "Black Fabric" material with a black base colour (any texture multiplied
to black), and the floor is a low-contrast concrete texture tiled every 0.4 m, glossy
enough to show a glare that moves with the camera. This copies the build to
<sim root>/<variant>_textured and, in its sharedassets0.assets:

  - replaces the wall texture (metal_0041_color_4k) with random black/white squares and
    circles, base colour white, one texture tile per `--wall-tile` metres;
  - replaces the floor texture (diffuse) the same way, one tile per `--floor-tile` metres,
    and sets the floor's smoothness to 0 (no view-dependent glare).

    python scripts/autodrive_texture_sim.py [--variant practice] [--root <sim dir>]
    # then SIM_VARIANT=practice_textured in sim.env, supervisorctl restart autodrive_sim

Needs UnityPy and Pillow (`pip install UnityPy pillow`), kept out of pilot_core's
dependencies. Only for sim experiments: benchmark results use the stock build.
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np

# measured on the stock 2026-iros meshes (Porto Track / the floor cube)
WALL_UV_PER_M = 39.4          # track mesh UV units per metre
FLOOR_SIZE_M = 1000.0         # floor cube scale; its UVs span 0..1


def pattern(size: int, tile_m: float, shape_m: tuple[float, float], fill: float, seed: int):
    """Random, non-overlapping-ish black squares and circles on white (RGB, uint8).
    The tile wraps, so shapes cut by an edge continue on the opposite side."""
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(seed)
    px_per_m = size / tile_m
    img = Image.new("RGB", (size, size), (235, 235, 235))
    d = ImageDraw.Draw(img)
    area, target = 0.0, fill * size * size
    while area < target:
        r = rng.uniform(*shape_m) * px_per_m / 2
        cx, cy = rng.uniform(0, size, 2)
        grey = int(rng.choice([15, 15, 60, 110]))      # mostly black, some greys
        draw = d.rectangle if rng.random() < 0.5 else d.ellipse
        for dx in (-size, 0, size):                      # wrapped copies
            for dy in (-size, 0, size):
                draw([cx + dx - r, cy + dy - r, cx + dx + r, cy + dy + r], fill=(grey,) * 3)
        area += (2 * r) ** 2 * 0.9
    return img


def set_props(mat_tree: dict, colors: dict, floats: dict, tex_scale: dict) -> None:
    props = mat_tree["m_SavedProperties"]
    for k, v in props["m_Colors"]:
        if k in colors:
            v.update(dict(zip("rgba", colors[k])))
    for i, (k, v) in enumerate(props["m_Floats"]):
        if k in floats:
            props["m_Floats"][i] = (k, floats[k])
    for k, v in props["m_TexEnvs"]:
        if k in tex_scale:
            v["m_Scale"] = {"x": tex_scale[k], "y": tex_scale[k]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default="/workspaces/autodrive_sim")
    ap.add_argument("--variant", default="practice")
    ap.add_argument("--wall-tile", type=float, default=1.0, help="m of wall per texture tile")
    ap.add_argument("--floor-tile", type=float, default=1.5, help="m of floor per texture tile")
    ap.add_argument("--size", type=int, default=1024, help="texture size in pixels")
    ap.add_argument("--preview", help="also write the two patterns to this directory")
    a = ap.parse_args()

    import UnityPy

    src = os.path.join(a.root, a.variant)
    dst = os.path.join(a.root, a.variant + "_textured")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    data = os.path.join(dst, "autodrive_simulator", "Data")
    path = os.path.join(data, "sharedassets0.assets")

    wall = pattern(a.size, a.wall_tile, (0.06, 0.16), 0.35, seed=1)
    floor = pattern(a.size, a.floor_tile, (0.10, 0.25), 0.30, seed=2)
    if a.preview:
        os.makedirs(a.preview, exist_ok=True)
        wall.save(os.path.join(a.preview, "wall_pattern.png"))
        floor.save(os.path.join(a.preview, "floor_pattern.png"))

    env = UnityPy.load(path)
    done = set()
    for o in env.objects:
        if o.type.name == "Texture2D":
            t = o.read()
            if t.m_Name in ("metal_0041_color_4k", "diffuse"):
                t.set_image((wall if t.m_Name == "metal_0041_color_4k" else floor).convert("RGBA"),
                            target_format=4, mipmap_count=1)        # RGBA32, no mips
                t.save()
                done.add(t.m_Name)
        elif o.type.name == "Material":
            tree = o.read_typetree()
            if tree["m_Name"] == "Black Fabric":
                set_props(tree, {"_BaseColor": (1, 1, 1, 1), "_Color": (1, 1, 1, 1)},
                          {"_Smoothness": 0.0, "_Metallic": 0.0},
                          {"_BaseColorMap": 1 / (a.wall_tile * WALL_UV_PER_M), "_MainTex": 1 / (a.wall_tile * WALL_UV_PER_M)})
                o.save_typetree(tree)
                done.add("Black Fabric")
            elif tree["m_Name"] == "Concrete pattern 03":
                set_props(tree, {}, {"_Smoothness": 0.0, "_Metallic": 0.0, "_NormalScale": 0.0},
                          {"_BaseColorMap": FLOOR_SIZE_M / a.floor_tile, "_MainTex": FLOOR_SIZE_M / a.floor_tile})
                o.save_typetree(tree)
                done.add("Concrete pattern 03")
    missing = {"metal_0041_color_4k", "diffuse", "Black Fabric", "Concrete pattern 03"} - done
    if missing:
        raise SystemExit(f"not found in {path}: {sorted(missing)}")
    out = env.file.save()       # before opening for write: unchanged objects are read lazily from path
    with open(path, "wb") as f:
        f.write(out)
    print(f"wrote {dst} (SIM_VARIANT={a.variant}_textured)")


if __name__ == "__main__":
    main()
