"""Command line entry point: generate candidate tiles for a set of crops from a JSON config.

    python run.py --config configs/example_portrait.json

The config format is described in crop_config.py, and the crop -> tile layout in
crop_layout.py. Tile sizes and their arrangement come from the config's `crops`; the
combination images come out at the config's `width` x `height`.

Use --dry-run to check the config and the layout without loading Stable Diffusion.
"""

import argparse
import gc
import os

import torch

from crop_config import load_crop_config
from crop_layout import build_latents, check_max_width, describe
from crop_output import save_all_combinations, save_run_meta, save_tiles

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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
    parser.add_argument('--cfg-scale', type=float, default=7.5,
                        help="Classifier free guidance scale. Default: 7.5")
    parser.add_argument('--scheduler', default='ddpm', choices=['ddpm', 'ddim', 'euler'],
                        help="Scheduler. Default: ddpm")

    parser.add_argument('--max-width', type=int, default=32,
                        help="Context size w, in latent pixels, used for the tiling constraint. "
                             "Must be a multiple of 4. Default: 32")
    parser.add_argument('--max-replica-width', type=int, default=5,
                        help="Width, in latent pixels, of the similarity constraint band. Default: 5")

    parser.add_argument('--max-combos', type=int, default=0,
                        help="Cap on how many candidate combinations to save. 0 means all. Default: 0")
    parser.add_argument('--dry-run', action='store_true',
                        help="Print the crop layout and exit, without loading the diffusion model.")
    parser.add_argument('--show', action='store_true',
                        help="Also display the first combination with matplotlib when finished.")

    return parser.parse_args()


def main():
    args = parse_args()
    check_max_width(args.max_width)

    cfg = load_crop_config(args.config)
    latents_arr, placed_crops = build_latents(cfg)

    print(describe(cfg, latents_arr, placed_crops))

    if args.dry_run:
        print("--dry-run: stopping before loading the model.")
        return

    # Only now do we pay for downloading / loading Stable Diffusion onto the GPU.
    import crop_latent_handler
    from model import SDLatentTiling

    # Crops have different heights, which the stock LatentHandler.tile mishandles.
    crop_latent_handler.install()

    torch.cuda.empty_cache()
    gc.collect()

    model = SDLatentTiling(scheduler=args.scheduler)
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
    save_tiles(latents_arr, placed_crops, out_dir)
    combo_paths = save_all_combinations(cfg, latents_arr, placed_crops, out_dir,
                                        max_combos=args.max_combos)
    save_run_meta(cfg, args, placed_crops, out_dir)

    if args.show and combo_paths:
        import matplotlib.pyplot as plt
        from PIL import Image
        plt.imshow(Image.open(combo_paths[0]))
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    main()
