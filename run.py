"""Command line entry point: generate candidate tiles for a set of crops from a JSON config.

    python run.py --config configs/example_portrait.json

The config format is described in crop_config.py, and the crop -> tile layout in
crop_layout.py. Tile sizes come from either FORCE_SQUARE_TILE_SIZE below or the config's
crop boxes; that choice also decides whether the crops are chained in the order the config
lists them or in the order their boxes stack.

Which denoiser is used is the USE_PIXELDIT macro below: NVIDIA's PixelDiT by default, Stable
Diffusion 1.5 when it is off. Nothing else about running the script changes between them.

Use --dry-run to check the config and the layout without loading the model.
"""

import argparse
import gc
import os

import torch

from crop_config import load_crop_config
from crop_layout import build_latents, check_max_width, describe
from crop_output import save_all_combinations, save_run_meta, save_tiles

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Which denoiser the run uses. True is NVIDIA's PixelDiT (pixel space, Gemma conditioned, flow DPM-Solver++);
# False is the original Stable Diffusion 1.5 path. Only the defaults below and the class picked in main()
# depend on it -- the config, the crop layout and the outputs are the same either way.
USE_PIXELDIT = True

# Generate every tile as a square of this many pixels, ignoring the config's crop boxes
# entirely: the crops are chained in the order the config lists them, so any arrangement
# of boxes works. The combination images are the tiles stacked, so they are this wide and
# this tall times the number of crops, not the config's width x height.
#
# Each model runs at the size it was released for: the PixelDiT checkpoint is the 1024px one
# (nvidia/PixelDiT-1300M-1024px), Stable Diffusion 1.5 is a 512px model.
#
# Set this to None to use each crop's own size from the config instead, in which case the
# combination images come out at exactly the config's width x height.
FORCE_SQUARE_TILE_SIZE = 1024 if USE_PIXELDIT else 512

# Defaults that are not comparable between the two models, so each one brings its own.
#   max width: PixelDiT counts 16 pixel patches, Stable Diffusion counts latent pixels (8 image pixels each).
#              Stable Diffusion's 32 latent pixels is a 256 pixel context on each side, which on its 512 pixel
#              tile is half the tile again and smears the seams badly. 4 patches is a 64 pixel context, which
#              is enough for the seam without taking over the tile.
#   max replica width: the similarity band, in the same units. Stable Diffusion's 5 latent pixels is 40 image
#              pixels, so 2 patches (32 pixels) is the nearest equivalent and still fits inside the context.
#   cfg scale: PixelDiT is sampled far lower than Stable Diffusion; the checkpoint's own default is 2.75.
DEFAULT_MAX_WIDTH = 4 if USE_PIXELDIT else 32
DEFAULT_MAX_REPLICA_WIDTH = 2 if USE_PIXELDIT else 5
DEFAULT_CFG_SCALE = 4.5 if USE_PIXELDIT else 7.5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True,
                        help="Path to the JSON config holding crops / tile_prompts / tiles_per_crop.")
    parser.add_argument('--out', default='outputs',
                        help="Output directory. Results go into <out>/<config name>_<prefix>/. "
                             "Default: outputs")

    parser.add_argument('--steps', type=int, default=40, help="Number of denoising steps. Default: 40")
    parser.add_argument('--seed', type=int, default=151, help="Random seed. Default: 151")
    parser.add_argument('--cfg-scale', type=float, default=DEFAULT_CFG_SCALE,
                        help=f"Classifier free guidance scale. Default: {DEFAULT_CFG_SCALE}")
    parser.add_argument('--scheduler', default='ddpm', choices=['ddpm', 'ddim', 'euler'],
                        help="Scheduler. Stable Diffusion only; PixelDiT always samples with flow "
                             "DPM-Solver++. Default: ddpm")

    parser.add_argument('--max-width', type=int, default=DEFAULT_MAX_WIDTH,
                        help=f"Context size w used for the tiling constraint, in 16 pixel patches with "
                             f"PixelDiT and in latent pixels (a multiple of 4) with Stable Diffusion. "
                             f"Default: {DEFAULT_MAX_WIDTH}")
    parser.add_argument('--max-replica-width', type=int, default=DEFAULT_MAX_REPLICA_WIDTH,
                        help=f"Width of the similarity constraint band, in the same units as --max-width. "
                             f"Default: {DEFAULT_MAX_REPLICA_WIDTH}")

    parser.add_argument('--combos', action=argparse.BooleanOptionalAction, default=True,
                        help="Save one full size image per combination of candidates, on top of the separate "
                             "tiles. --no-combos writes the tiles only, which is much faster when a config has "
                             "many candidates (tiles_per_crop ** num_crops combinations). Default: --combos")
    parser.add_argument('--max-combos', type=int, default=0,
                        help="Cap on how many candidate combinations to save. 0 means all. Ignored with "
                             "--no-combos. Default: 0")
    parser.add_argument('--dry-run', action='store_true',
                        help="Print the crop layout and exit, without loading the diffusion model.")
    parser.add_argument('--show', action='store_true',
                        help="Also display the first combination with matplotlib when finished. Needs --combos.")

    return parser.parse_args()


def main():
    args = parse_args()
    check_max_width(args.max_width, patch_units=USE_PIXELDIT)

    cfg = load_crop_config(args.config)
    latents_arr, layout = build_latents(cfg, force_square_size=FORCE_SQUARE_TILE_SIZE)

    print(describe(cfg, latents_arr, layout))

    if args.dry_run:
        print("--dry-run: stopping before loading the model.")
        return

    # Only now do we pay for downloading / loading the model onto the GPU.
    import crop_latent_handler
    import model as model_module

    # Crops have different heights, which the stock LatentHandler.tile mishandles.
    crop_latent_handler.install()

    torch.cuda.empty_cache()
    gc.collect()

    tiling_class = model_module.PixelDiTLatentTiling if USE_PIXELDIT else model_module.SDLatentTiling
    model = tiling_class(scheduler=args.scheduler)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    latents_arr = model(latents_arr=latents_arr,
                        inference_steps=args.steps,
                        seed=args.seed,
                        cfg_scale=args.cfg_scale,
                        max_width=args.max_width,
                        max_replica_width=args.max_replica_width,
                        device=device)

    torch.cuda.empty_cache()
    gc.collect()

    out_dir = os.path.join(args.out, cfg.output_name)
    os.makedirs(out_dir, exist_ok=True)
    save_tiles(latents_arr, layout, out_dir)
    combo_paths = []
    if args.combos:
        combo_paths = save_all_combinations(latents_arr, layout, out_dir,
                                            max_combos=args.max_combos)
    else:
        print("--no-combos: skipping the combination images, the tiles are in tiles/")
    save_run_meta(cfg, args, layout, out_dir)

    if args.show and not combo_paths:
        print("--show needs a combination image, and none were saved.")
    elif args.show:
        import matplotlib.pyplot as plt
        from PIL import Image
        plt.imshow(Image.open(combo_paths[0]))
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    main()
