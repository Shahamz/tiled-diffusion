"""Turning a CropConfig into the array of LatentClass tiles the model expects.

--------------------------------------------------------------------------------
Candidates
--------------------------------------------------------------------------------
A *crop* is a region of the final image. A *candidate* is one small image that could
fill that region. This repo only knows about tiles and how their sides connect, so we
express "candidate" purely through the connection descriptors:

    all candidates of a crop are given the IDENTICAL side_id / side_dir pair,
    and differ only in their prompt.

Because `utils.generate_graph_groups` matches sides by (side_id, opposite side_dir) and
`LatentHandler.apply_similarity_constraint` then forces every matched side to carry the
same latent band, giving two candidates the same descriptors is exactly what makes them
interchangeable: either one can be dropped into the crop and the seam with the
neighbouring crop is the same.

All candidates of a crop are also generated at the same size -- the crop's size.

--------------------------------------------------------------------------------
Layout
--------------------------------------------------------------------------------
The layout is read from the crops' positions in the config: the crops are sorted by
their y coordinate, top to bottom, and each one is connected to the one below it.
Only a vertical stack is supported, so every crop must share the same x and width;
anything else is rejected rather than guessed at.

Sides are indexed [Right, Left, Up, Down] (see config.py). With the crops in top to
bottom order, junction `i` sits between crop `i-1` and crop `i`:

    top       side_id=[None, None, None, 1   ]   side_dir=[None, None, None,  'cw' ]
      -- junction 1 --
              side_id=[None, None, 1,    2   ]   side_dir=[None, None, 'ccw', 'cw' ]
      -- junction 2 --
              side_id=[None, None, i,    i+1 ]   side_dir=[None, None, 'ccw', 'cw' ]
      ...
    bottom    side_id=[None, None, N-1,  None]   side_dir=[None, None, 'ccw', None ]

Left and right sides are left unconstrained. With a single crop nothing connects to
anything and each tile is an ordinary unconstrained generation.

--------------------------------------------------------------------------------
Size snapping
--------------------------------------------------------------------------------
Stable Diffusion's UNet downsamples by 8, so a latent side must be divisible by 8 and a
pixel side by 64. Crop sizes from the config are rarely multiples of 64 (208, 432, ...),
so each crop is generated at the nearest multiple of 64. The generated tile is resized
back to the crop's exact size when the final combination image is composed, which is why
the combinations come out at exactly the config's width x height.
"""

from latent_class import LatentClass

# Side indices, as used by LatentClass.side_id / side_dir. Mirrors config.DIRECTION_STRING_TO_INDEX.
RIGHT, LEFT, UP, DOWN = 0, 1, 2, 3

# Pixel sizes must be a multiple of this for the UNet's three downsampling stages.
SIZE_GRANULARITY = 64


class PlacedCrop:
    """A crop, its position in the top-to-bottom chain, and the size we generate it at."""

    def __init__(self, crop, position, gen_width, gen_height):
        self.crop = crop                # the config's Crop, with its exact x/y/width/height
        self.position = position        # 0 = topmost, num_crops - 1 = bottommost
        self.gen_width = gen_width      # snapped to a multiple of 64
        self.gen_height = gen_height
        self.tile_indices = []          # positions of this crop's candidates in latents_arr

    @property
    def index(self):
        """Index into the config's crops / tile_prompts lists."""
        return self.crop.index

    @property
    def was_snapped(self):
        return (self.gen_width, self.gen_height) != (self.crop.width, self.crop.height)


def snap_to_granularity(value):
    """Nearest multiple of 64, never zero."""
    return max(SIZE_GRANULARITY, int(round(value / SIZE_GRANULARITY)) * SIZE_GRANULARITY)


def order_crops_top_to_bottom(cfg):
    """Sort the config's crops into a vertical chain, rejecting anything that is not one."""
    xs = {crop.x for crop in cfg.crops}
    widths = {crop.width for crop in cfg.crops}
    if len(xs) > 1 or len(widths) > 1:
        raise ValueError(
            "Only a vertical stack of crops is supported: every crop must share the same 'x' and "
            f"'width'. Got x values {sorted(xs)} and widths {sorted(widths)}. "
            "Side by side crops would need left/right connections, which this script does not build."
        )

    ys = [crop.y for crop in cfg.crops]
    if len(set(ys)) != len(ys):
        raise ValueError(f"Two or more crops share the same 'y' ({sorted(ys)}), so their top to "
                         f"bottom order is ambiguous.")

    ordered = sorted(cfg.crops, key=lambda crop: crop.y)
    return [PlacedCrop(crop, position,
                       snap_to_granularity(crop.width),
                       snap_to_granularity(crop.height))
            for position, crop in enumerate(ordered)]


def crop_side_descriptors(position, num_crops):
    """Return the (side_id, side_dir) pair shared by every candidate at chain `position`.

    The crop's UP side carries junction `position`, its DOWN side junction `position + 1`.
    The two ends of the chain have nothing above / below them, so those sides stay None.
    A junction connects because the two sides facing each other carry the same id with
    opposite orientations: the upper crop's DOWN is 'cw', the lower crop's UP is 'ccw'.
    """
    side_id = [None, None, None, None]
    side_dir = [None, None, None, None]

    has_crop_above = position > 0
    has_crop_below = position < num_crops - 1

    if has_crop_above:
        side_id[UP] = position
        side_dir[UP] = 'ccw'

    if has_crop_below:
        side_id[DOWN] = position + 1
        side_dir[DOWN] = 'cw'

    return side_id, side_dir


def build_latents(cfg):
    """Build every candidate tile for every crop.

    Returns (latents_arr, placed_crops). `latents_arr` is the flat list handed to
    SDLatentTiling, in top-to-bottom crop order; each PlacedCrop records which positions
    in that list are its own candidates, in `tile_indices`.
    """
    placed_crops = order_crops_top_to_bottom(cfg)
    latents_arr = []

    for placed in placed_crops:
        side_id, side_dir = crop_side_descriptors(placed.position, cfg.num_crops)

        for candidate_idx in range(cfg.tiles_per_crop):
            latent = LatentClass(
                prompt=cfg.prompt(placed.index, candidate_idx),
                negative_prompt=cfg.negative_prompt(placed.index, candidate_idx),
                height=placed.gen_height,
                width=placed.gen_width,
                # Every candidate of this crop gets its own copy of the same descriptors.
                side_id=list(side_id),
                side_dir=list(side_dir),
            )
            placed.tile_indices.append(len(latents_arr))
            latents_arr.append(latent)

    return latents_arr, placed_crops


def check_max_width(max_width):
    """The tiling padding is added to the latent, so it too must keep it divisible by 8."""
    if max_width < 4 or (2 * max_width) % 8 != 0:
        raise ValueError(f"--max-width must be a positive multiple of 4, got {max_width}. "
                         f"The padding it adds ({2 * max_width} latent pixels) has to keep the "
                         f"latent divisible by 8.")


def describe(cfg, latents_arr, placed_crops):
    """Return a human readable table of the layout and wiring, for --dry-run."""
    lines = []
    lines.append(f"image {cfg.image_width}x{cfg.image_height}, "
                 f"{cfg.num_crops} crops x {cfg.tiles_per_crop} candidates = {len(latents_arr)} tiles")
    lines.append("")
    lines.append("Crops, top to bottom:")
    lines.append(f"  {'pos':>3}  {'cfg':>3}  {'box (x,y,w,h)':<24}  {'generated at':<14}  note")
    for placed in placed_crops:
        crop = placed.crop
        box = f"({crop.x},{crop.y}) {crop.width}x{crop.height}"
        gen = f"{placed.gen_width}x{placed.gen_height}"
        note = "snapped to a multiple of 64" if placed.was_snapped else ""
        lines.append(f"  {placed.position:>3}  {placed.index:>3}  {box:<24}  {gen:<14}  {note}")

    covered = sum(placed.crop.height for placed in placed_crops)
    if covered != cfg.image_height:
        lines.append(f"  note: the crops cover {covered}px of the image's {cfg.image_height}px height")

    lines.append("")
    lines.append(f"{'idx':>4}  {'pos':>3}  {'cfg':>3}  {'cand':>4}  "
                 f"{'side_id (R,L,U,D)':<22}  {'side_dir (R,L,U,D)':<30}  prompt")
    lines.append("-" * 130)

    for placed in placed_crops:
        for candidate_idx, flat_idx in enumerate(placed.tile_indices):
            latent = latents_arr[flat_idx]
            lines.append(
                f"{flat_idx:>4}  {placed.position:>3}  {placed.index:>3}  {candidate_idx:>4}  "
                f"{str(latent.side_id):<22}  {str(latent.side_dir):<30}  {latent.prompt}"
            )
        lines.append("")

    return "\n".join(lines)
