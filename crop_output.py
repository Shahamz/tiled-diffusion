"""Writing the generated tiles to disk.

Two kinds of output:

  tiles/   one PNG per candidate tile, at the size it was generated at (a multiple of 64)
  combos/  one PNG per way of picking a candidate for every crop, composed at exactly the
           config's width x height: each chosen tile is resized to its crop's exact size
           and pasted at the crop's (x, y)

The combos are the point of the experiment: if candidates of a crop really are
interchangeable, every combo should show the same seams.
"""

import itertools
import json
import os

import numpy as np
from PIL import Image

# Colour the combination canvas starts as, so any area no crop covers is obvious.
UNCOVERED_COLOUR = (0, 0, 0)


def _to_image(latent):
    """LatentClass.image is an HWC float array in [0, 1]."""
    return Image.fromarray((np.asarray(latent.image) * 255.0).clip(0, 255).astype(np.uint8))


def save_tiles(latents_arr, placed_crops, out_dir):
    """Write every candidate tile, named by its index in the config. Returns the paths written."""
    tiles_dir = os.path.join(out_dir, 'tiles')
    os.makedirs(tiles_dir, exist_ok=True)

    written = []
    for placed in placed_crops:
        for candidate_idx, flat_idx in enumerate(placed.tile_indices):
            path = os.path.join(tiles_dir, f"crop{placed.index:02d}_cand{candidate_idx:02d}.png")
            _to_image(latents_arr[flat_idx]).save(path)
            written.append(path)

    print(f"Wrote {len(written)} tiles to {tiles_dir}")
    return written


def compose_combination(latents_arr, placed_crops, choice, image_width, image_height):
    """Build one full size image from one candidate per crop.

    `choice[position]` is the candidate index to use for the crop at that chain position.
    Each tile is resized from its generated (multiple of 64) size back to the crop's exact
    size, so the result is exactly image_width x image_height.
    """
    canvas = Image.new('RGB', (image_width, image_height), UNCOVERED_COLOUR)

    for placed, candidate_idx in zip(placed_crops, choice):
        tile = _to_image(latents_arr[placed.tile_indices[candidate_idx]])
        crop = placed.crop
        if tile.size != (crop.width, crop.height):
            tile = tile.resize((crop.width, crop.height), Image.LANCZOS)
        canvas.paste(tile, (crop.x, crop.y))

    return canvas


def save_all_combinations(cfg, latents_arr, placed_crops, out_dir, max_combos=0):
    """Write one full size image per combination of candidates.

    There are `tiles_per_crop ** num_crops` combinations, which grows fast.
    `max_combos` caps how many are written (0 means no cap).
    """
    combos_dir = os.path.join(out_dir, 'combos')
    os.makedirs(combos_dir, exist_ok=True)

    candidates_per_crop = [len(placed.tile_indices) for placed in placed_crops]
    total = 1
    for count in candidates_per_crop:
        total *= count

    print(f"{total} combination(s) of candidates ({' x '.join(str(c) for c in candidates_per_crop)}), "
          f"each {cfg.image_width}x{cfg.image_height}")
    if max_combos and total > max_combos:
        print(f"  --max-combos is {max_combos}, so only the first {max_combos} will be written")

    written = []
    for choice in itertools.product(*[range(count) for count in candidates_per_crop]):
        if max_combos and len(written) >= max_combos:
            break

        canvas = compose_combination(latents_arr, placed_crops, choice,
                                     cfg.image_width, cfg.image_height)
        # Name the combination by config crop index order, so it matches tile_prompts.
        by_config_index = [candidate_idx for _, candidate_idx
                           in sorted(zip([p.index for p in placed_crops], choice))]
        name = "combo_" + "-".join(f"{candidate_idx:02d}" for candidate_idx in by_config_index) + ".png"
        path = os.path.join(combos_dir, name)
        canvas.save(path)
        written.append(path)

    print(f"Wrote {len(written)} combination image(s) to {combos_dir}")
    return written


def save_run_meta(cfg, args, placed_crops, out_dir):
    """Dump the settings, prompts and sizes actually used, next to the images."""
    path = os.path.join(out_dir, 'run_meta.json')
    meta = {
        'args': vars(args),
        'image_width': cfg.image_width,
        'image_height': cfg.image_height,
        'num_crops': cfg.num_crops,
        'tiles_per_crop': cfg.tiles_per_crop,
        'tile_prompts': cfg.tile_prompts,
        'tile_negative_prompts': cfg.tile_negative_prompts,
        'crops': [
            {
                'config_index': placed.index,
                'chain_position': placed.position,
                'x': placed.crop.x, 'y': placed.crop.y,
                'width': placed.crop.width, 'height': placed.crop.height,
                'generated_width': placed.gen_width, 'generated_height': placed.gen_height,
            }
            for placed in placed_crops
        ],
    }
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {path}")
    return path
