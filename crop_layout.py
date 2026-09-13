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

All candidates of a crop are also generated at the same size.

--------------------------------------------------------------------------------
Layout
--------------------------------------------------------------------------------
The crops form a chain, each one connected to the next. Where that order comes from
depends on the sizing mode below: when tile sizes are forced the config's crop boxes say
nothing we need, so the order is simply the order they are listed in, which is already the
order `tile_prompts` is in. When sizes come from the config the boxes do matter, so the
crops are sorted by their y coordinate, top to bottom; that path supports a vertical stack
only, and rejects anything else rather than guessing at it.

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
Two sizing modes
--------------------------------------------------------------------------------
`build_latents` takes `force_square_size`, which run.py drives from one macro:

  force_square_size = N     every tile is generated as an N x N square and the crop boxes
                            in the config are ignored entirely, including their positions.
                            The final image is the tiles stacked in config order, N wide by
                            N * num_crops tall, so it is not the config's width x height.

  force_square_size = None  every tile is generated at its own crop's size and pasted
                            back into the crop's exact box, so the final image comes out
                            at exactly the config's width x height.

Stable Diffusion's UNet downsamples by 8, so a latent side must be divisible by 8 and a
pixel side by 64. Sizes coming from either mode are rounded to the nearest multiple of 64
for generation; in config-size mode the tile is resized back to the crop's exact size
when the final image is composed.
"""

from latent_class import LatentClass

# Side indices, as used by LatentClass.side_id / side_dir. Mirrors config.DIRECTION_STRING_TO_INDEX.
RIGHT, LEFT, UP, DOWN = 0, 1, 2, 3

# Pixel sizes must be a multiple of this for the UNet's three downsampling stages.
SIZE_GRANULARITY = 64


class PlacedCrop:
    """A crop, its position in the top-to-bottom chain, the size we generate it at,
    and the box it occupies in the final image."""

    def __init__(self, crop, position, gen_width, gen_height,
                 place_x, place_y, place_width, place_height):
        self.crop = crop                # the config's Crop, with its exact x/y/width/height
        self.position = position        # 0 = topmost, num_crops - 1 = bottommost
        self.gen_width = gen_width      # size we generate at, a multiple of 64
        self.gen_height = gen_height
        self.place_x = place_x          # where the finished tile goes in the final image
        self.place_y = place_y
        self.place_width = place_width
        self.place_height = place_height
        self.tile_indices = []          # positions of this crop's candidates in latents_arr

    @property
    def index(self):
        """Index into the config's crops / tile_prompts lists."""
        return self.crop.index

    @property
    def needs_resize(self):
        return (self.gen_width, self.gen_height) != (self.place_width, self.place_height)


class Layout:
    """The crops in chain order, plus the size of the image they compose into."""

    def __init__(self, placed_crops, canvas_width, canvas_height):
        self.placed_crops = placed_crops
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

    def __iter__(self):
        return iter(self.placed_crops)


def snap_to_granularity(value):
    """Nearest multiple of 64, never zero."""
    return max(SIZE_GRANULARITY, int(round(value / SIZE_GRANULARITY)) * SIZE_GRANULARITY)


def order_crops_top_to_bottom(cfg):
    """Sort the config's crops into a vertical chain, rejecting anything that is not one.

    Only used when tile sizes come from the config. When sizes are forced the boxes are
    ignored and the crops are chained in the order the config lists them.
    """
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

    return sorted(cfg.crops, key=lambda crop: crop.y)


def plan_layout(cfg, force_square_size=None):
    """Decide what size each crop is generated at and where its tile lands in the final image."""
    if force_square_size is not None:
        # Squares of a single size, stacked. Nothing about the config's boxes is used: the
        # crops are chained in the order they are listed, which is the order tile_prompts
        # is in, so any arrangement of boxes works here, side by side ones included.
        size = snap_to_granularity(force_square_size)
        placed_crops = [
            PlacedCrop(crop, position,
                       gen_width=size, gen_height=size,
                       place_x=0, place_y=position * size,
                       place_width=size, place_height=size)
            for position, crop in enumerate(cfg.crops)
        ]
        return Layout(placed_crops, canvas_width=size, canvas_height=size * len(cfg.crops))

    # Each crop at its own size, back in its own box. Here the boxes decide the order.
    ordered = order_crops_top_to_bottom(cfg)
    placed_crops = [
        PlacedCrop(crop, position,
                   gen_width=snap_to_granularity(crop.width),
                   gen_height=snap_to_granularity(crop.height),
                   place_x=crop.x, place_y=crop.y,
                   place_width=crop.width, place_height=crop.height)
        for position, crop in enumerate(ordered)
    ]
    return Layout(placed_crops, canvas_width=cfg.image_width, canvas_height=cfg.image_height)


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


def build_latents(cfg, force_square_size=None):
    """Build every candidate tile for every crop.

    Returns (latents_arr, layout). `latents_arr` is the flat list handed to SDLatentTiling,
    in top-to-bottom crop order; each PlacedCrop in the layout records which positions in
    that list are its own candidates, in `tile_indices`.
    """
    layout = plan_layout(cfg, force_square_size)
    latents_arr = []

    for placed in layout:
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

    return latents_arr, layout


def check_max_width(max_width):
    """The tiling padding is added to the latent, so it too must keep it divisible by 8."""
    if max_width < 4 or (2 * max_width) % 8 != 0:
        raise ValueError(f"--max-width must be a positive multiple of 4, got {max_width}. "
                         f"The padding it adds ({2 * max_width} latent pixels) has to keep the "
                         f"latent divisible by 8.")


def describe(cfg, latents_arr, layout):
    """Return a human readable table of the layout and wiring, for --dry-run."""
    lines = []
    lines.append(f"{cfg.num_crops} crops x {cfg.tiles_per_crop} candidates = {len(latents_arr)} tiles, "
                 f"composed into {layout.canvas_width}x{layout.canvas_height} "
                 f"(config image is {cfg.image_width}x{cfg.image_height})")
    lines.append("")
    lines.append("Crops, in chain order:")
    lines.append(f"  {'pos':>3}  {'cfg':>3}  {'config box (x,y,w,h)':<24}  "
                 f"{'generated':<12}  {'placed at (x,y,w,h)':<24}")
    for placed in layout:
        crop = placed.crop
        box = f"({crop.x},{crop.y}) {crop.width}x{crop.height}"
        gen = f"{placed.gen_width}x{placed.gen_height}"
        place = (f"({placed.place_x},{placed.place_y}) "
                 f"{placed.place_width}x{placed.place_height}")
        lines.append(f"  {placed.position:>3}  {placed.index:>3}  {box:<24}  {gen:<12}  {place:<24}")

    lines.append("")
    lines.append(f"{'idx':>4}  {'pos':>3}  {'cfg':>3}  {'cand':>4}  "
                 f"{'side_id (R,L,U,D)':<22}  {'side_dir (R,L,U,D)':<30}  prompt")
    lines.append("-" * 130)

    for placed in layout:
        for candidate_idx, flat_idx in enumerate(placed.tile_indices):
            latent = latents_arr[flat_idx]
            lines.append(
                f"{flat_idx:>4}  {placed.position:>3}  {placed.index:>3}  {candidate_idx:>4}  "
                f"{str(latent.side_id):<22}  {str(latent.side_dir):<30}  {latent.prompt}"
            )
        lines.append("")

    return "\n".join(lines)
