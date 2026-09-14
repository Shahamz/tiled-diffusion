"""
Tile attention rules for Mix_n_match.

After the split at late_prompting, one transformer sequence holds (in this order):

    [background prompt | tile prompts (crop-major) | tile tokens (k x tile patches) | background patches
     | crop-average keys]

The sample is k full images; image s holds tile s of every crop. The background crop (config "background_crop", the
region the background prompt won in the saliency, see saliency.py) is NOT split into tiles: its patches appear ONCE
in the sequence, taken from image 0, and the model copies their result back into every image, so all k images keep
the same background region. Without a background crop there are no background patches and nothing changes.
The crop-average keys exist only with attend_on_avg; they are appended to the keys/values (never to the queries)
and hold, at every tile patch location, the average of the k tiles of the crop covering it.

Who may attend whom (query -> key):

    query \\ key        | background | own tile  | own    | sibling tile | tile of     | average of  | background
                        | prompt     | prompt    | tile   | (same crop)  | other crop  | other crop  | patches
    background prompt   |    yes     |    no     |  yes   |     yes      |    yes      |    no       |    yes
    tile prompt (i,j)   |    no      |    yes    |  yes   |     no       |    no       |    no       |    no
    tile (i,j)          |    yes     |    yes    |  yes   |     no       | without avg | with avg    |    yes
    background patch    |    yes     |    no     |  no    |     no       | without avg | with avg    |    yes

So the background region and the tiles see each other (a tile sees the one shared background region; the
background sees the tile crops the way a tile sees another crop, i.e. their averages with attend_on_avg), while
the tile prompts stay blind to the background region. The pixel stage has no text; it uses the same rule on the
image tokens and crop-average keys.

Patch order (group_patches_by_crop): flex attention skips a 128 x 128 block of (query, key) pairs only when EVERY
pair in it is masked; a block with any allowed pair is computed in full and then masked. In raster order, crops
that share image rows (e.g. side-by-side crops) are interleaved, so almost no block can be skipped. With
group_patches_by_crop, the patches of every image are sorted by crop (raster order inside a crop) once at the model
input and put back at its output. All images use the same order, so every (crop, tile) group is contiguous. For
horizontal stripes the sort changes nothing and is skipped. A background crop always sorts, because its patches
must end up together, at the end of every image.

This file labels every token with (kind, crop, tile), builds the patch order and the "background once" selection,
turns the rule into a flex-attention BlockMask, and runs the attention.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

# Token kinds used to label every sequence position.
BACKGROUND_PROMPT_TOKEN = 0
TILE_PROMPT_TOKEN = 1
TILE_TOKEN = 2
CROP_AVERAGE_TOKEN = 3  # key-only token: the average of a crop's tiles at one patch location (attend_on_avg)
BACKGROUND_TOKEN = 4  # a patch of the background crop, kept once for all k images

# Crop/tile label of tokens that belong to no crop or no tile (background prompt, crop-average keys).
NO_INDEX = -1

# Compiled versions: compiled flex attention skips masked blocks without ever building the full score matrix,
# and compiled mask creation avoids materializing the full boolean mask.
compiled_flex_attention = torch.compile(flex_attention)
compiled_create_block_mask = torch.compile(create_block_mask)


@dataclass
class TileAttentionLayout:
    """Everything the transformer needs to run masked attention (after the split, or at the saliency step)."""

    num_tiles: int  # k: tiles per crop, which is also the number of images in the sample
    average_target: str | None  # "keys_values" or "hidden_states" when attend_on_avg is on, otherwise None
    joint_block_mask: BlockMask  # patch stage: prompts + image tokens (+ crop-average keys)
    pixel_block_mask: BlockMask | None  # pixel stage: image tokens (+ crop-average keys); None = full attention
    patch_order: torch.Tensor | None = None  # raster index of the patch at every image position; None = raster order
    patch_restore: torch.Tensor | None = None  # inverse of patch_order
    tile_patch_count: int = 0  # patches per image that belong to tile crops; they come first in the patch order
    background_patch_count: int = 0  # patches of the background crop, kept once for all images
    saliency_probe: object | None = None  # saliency.SaliencyProbe at the saliency step, otherwise None


def tile_mask_mod(labels, other_crop_kind):
    """
    Builds the flex-attention mask function that encodes the rule table at the top of this file.

    Queries are the first positions of the key sequence (crop-average keys only exist at the end), so one label
    tensor serves both.

    Args:
        labels: [3, key_length] integer tensor with the (kind, crop, tile) of every key position.
        other_crop_kind: TILE_TOKEN (tiles see other crops' tiles) or CROP_AVERAGE_TOKEN (they see the averages).

    Returns:
        mask_mod(batch, head, query_index, key_index) -> bool tensor, True where attention is allowed.
    """
    kinds, crops, tiles = labels

    def mask_mod(batch, head, query_index, key_index):
        query_kind, key_kind = kinds[query_index], kinds[key_index]
        same_crop = crops[query_index] == crops[key_index]
        same_tile = same_crop & (tiles[query_index] == tiles[key_index])
        key_in_own_tile = same_tile & ((key_kind == TILE_PROMPT_TOKEN) | (key_kind == TILE_TOKEN))

        background_query_allowed = (query_kind == BACKGROUND_PROMPT_TOKEN) & (
            (key_kind == BACKGROUND_PROMPT_TOKEN) | (key_kind == TILE_TOKEN) | (key_kind == BACKGROUND_TOKEN)
        )
        tile_prompt_query_allowed = (query_kind == TILE_PROMPT_TOKEN) & key_in_own_tile
        tile_query_allowed = (query_kind == TILE_TOKEN) & (
            (key_kind == BACKGROUND_PROMPT_TOKEN) | key_in_own_tile | (key_kind == BACKGROUND_TOKEN)
            | (~same_crop & (key_kind == other_crop_kind))
        )
        # The background region sees the tile crops the way a tile sees another crop: their averages with
        # attend_on_avg, the tile tokens themselves without it.
        background_patch_query_allowed = (query_kind == BACKGROUND_TOKEN) & (
            (key_kind == BACKGROUND_PROMPT_TOKEN) | (key_kind == BACKGROUND_TOKEN) | (key_kind == other_crop_kind)
        )
        return (background_query_allowed | tile_prompt_query_allowed | tile_query_allowed
                | background_patch_query_allowed)

    return mask_mod


def build_patch_order(patch_crops):
    """
    Sorts the patches of an image by crop, keeping raster order inside a crop.

    Args:
        patch_crops: [L] tensor with the crop index of every patch, raster order. The background crop has the
            largest index, so its patches end up last, which is what keep_background_once expects.

    Returns:
        (patch_order, patch_restore): [L] tensors, or (None, None) when raster order is already sorted.
    """
    patch_order = torch.argsort(patch_crops, stable=True)
    if torch.equal(patch_order, torch.arange(len(patch_order), device=patch_order.device)):
        return None, None
    return patch_order, torch.argsort(patch_order)


def build_token_labels(tile_patch_crops, background_patch_count, num_crops, num_tiles, prompt_length, attend_on_avg):
    """
    Labels every position of the patch-stage (joint) and pixel-stage sequences with (kind, crop, tile).

    Args:
        tile_patch_crops: [tile patches] tensor with the crop index of every tile-crop patch, in model order.
        background_patch_count: patches of the background crop (0 without one).
        num_crops: n, the number of tile crops (a crop may cover no patch).
        num_tiles: k, tiles per crop.
        prompt_length: number of tokens per prompt.
        attend_on_avg: whether crop-average keys are appended.

    Returns:
        (joint_labels, pixel_labels): [3, joint key length] and [3, pixel key length] integer tensors.
    """
    device = tile_patch_crops.device
    tile_patch_count = tile_patch_crops.numel()

    # Text: prompt 0 is the background prompt, prompt 1 + crop * k + tile is that tile's prompt.
    prompt_indices = torch.arange(1 + num_crops * num_tiles, device=device).repeat_interleave(prompt_length)
    is_tile_prompt = prompt_indices > 0
    text_labels = torch.stack([
        torch.where(is_tile_prompt, TILE_PROMPT_TOKEN, BACKGROUND_PROMPT_TOKEN),
        torch.where(is_tile_prompt, (prompt_indices - 1) // num_tiles, NO_INDEX),
        torch.where(is_tile_prompt, (prompt_indices - 1) % num_tiles, NO_INDEX),
    ])

    # Image s holds tile s of every crop, so a patch's crop comes from the crop map and its tile is s.
    image_labels = torch.stack([
        torch.full((num_tiles * tile_patch_count,), TILE_TOKEN, device=device),
        tile_patch_crops.repeat(num_tiles),
        torch.arange(num_tiles, device=device).repeat_interleave(tile_patch_count),
    ])
    if background_patch_count:
        image_labels = torch.cat([image_labels, torch.stack([
            torch.full((background_patch_count,), BACKGROUND_TOKEN, device=device),
            torch.full((background_patch_count,), NO_INDEX, device=device),
            torch.full((background_patch_count,), NO_INDEX, device=device),
        ])], dim=1)
    if attend_on_avg:
        image_labels = torch.cat([image_labels, torch.stack([
            torch.full((tile_patch_count,), CROP_AVERAGE_TOKEN, device=device),
            tile_patch_crops,
            torch.full((tile_patch_count,), NO_INDEX, device=device),
        ])], dim=1)
    return torch.cat([text_labels, image_labels], dim=1), image_labels


def build_tile_attention_layout(crop_map, num_crops, num_tiles, prompt_length, attend_on_avg, average_target,
                                group_patches_by_crop, device):
    """
    Builds the patch order, the "background once" selection and the block masks of both transformer stages.

    Args:
        crop_map: [patch rows, patch cols] array of crop indices; num_crops marks the background crop's patches.
        num_crops: n, the number of tile crops (a crop may cover no patch).
        num_tiles: k, tiles per crop.
        prompt_length: number of tokens per prompt.
        attend_on_avg: whether tiles attend other crops through their average.
        average_target: "keys_values" or "hidden_states" (ignored when attend_on_avg is false).
        group_patches_by_crop: True to sort every image's patches by crop, False for raster order (a background
            crop always sorts, so its patches end up together).
        device: torch device for the masks.

    Returns:
        TileAttentionLayout.
    """
    patch_crops = torch.as_tensor(crop_map, device=device).flatten()
    background_patch_count = int((patch_crops == num_crops).sum())
    patch_order, patch_restore = (build_patch_order(patch_crops) if group_patches_by_crop or background_patch_count
                                  else (None, None))
    if patch_order is not None:
        patch_crops = patch_crops[patch_order]
    tile_patch_count = patch_crops.numel() - background_patch_count

    joint_labels, pixel_labels = build_token_labels(
        patch_crops[:tile_patch_count], background_patch_count, num_crops, num_tiles, prompt_length, attend_on_avg
    )
    other_crop_kind = CROP_AVERAGE_TOKEN if attend_on_avg else TILE_TOKEN
    num_average_keys = tile_patch_count if attend_on_avg else 0

    block_masks = []
    for labels in (joint_labels, pixel_labels):
        key_length = labels.shape[1]
        block_masks.append(compiled_create_block_mask(
            tile_mask_mod(labels, other_crop_kind), None, None, key_length - num_average_keys, key_length, device=device
        ))
    return TileAttentionLayout(
        num_tiles, average_target if attend_on_avg else None, *block_masks, patch_order, patch_restore,
        tile_patch_count, background_patch_count,
    )


def keep_background_once(tokens, layout, num_images):
    """
    Joins the images into one token sequence, keeping the background crop's patches only once (from image 0).

    Args:
        tokens: [batch * num_images, patches, ...] tokens of every image, already in the layout's patch order.
        layout: TileAttentionLayout.
        num_images: S, the number of images in the sample.

    Returns:
        [batch, num_images * tile patches (+ background patches), ...] tensor.
    """
    per_image = tokens.view(tokens.shape[0] // num_images, num_images, *tokens.shape[1:])
    tile_tokens = per_image[:, :, :layout.tile_patch_count].flatten(1, 2)
    if not layout.background_patch_count:
        return tile_tokens
    return torch.cat([tile_tokens, per_image[:, 0, layout.tile_patch_count:]], dim=1)


def restore_all_images(tokens, layout, num_images):
    """
    Undoes keep_background_once: every image gets the tile patches it owns and a copy of the background patches.

    Args:
        tokens: [batch, kept tokens, ...] tensor.
        layout: TileAttentionLayout.
        num_images: S, the number of images in the sample.

    Returns:
        [batch * num_images, patches, ...] tensor, in the layout's patch order.
    """
    batch, tile_count = tokens.shape[0], layout.tile_patch_count
    tile_tokens = tokens[:, :num_images * tile_count].view(batch, num_images, tile_count, *tokens.shape[2:])
    if not layout.background_patch_count:
        return tile_tokens.flatten(0, 1)
    background = tokens[:, num_images * tile_count:]
    background = background.unsqueeze(1).expand(batch, num_images, *background.shape[1:])
    return torch.cat([tile_tokens, background], dim=2).flatten(0, 1)


def keep_positions(positions, layout, num_images):
    """
    The RoPE positions of the kept token sequence: the tile patches once per image, then the background patches.

    Args:
        positions: [patches, dim] positions in the layout's patch order.
        layout: TileAttentionLayout.
        num_images: S, the number of images in the sample.

    Returns:
        [kept tokens, dim] tensor.
    """
    tile_positions = positions[:layout.tile_patch_count].repeat(num_images, 1)
    if not layout.background_patch_count:
        return tile_positions
    return torch.cat([tile_positions, positions[layout.tile_patch_count:]])


def attention(query, key, value, block_mask):
    """
    Runs multi-head attention, masked by the tile rules when a block mask is given.

    Args:
        query: [batch, heads, query length, head dim].
        key, value: [batch, heads, key length, head dim].
        block_mask: BlockMask from a TileAttentionLayout, or None for full attention (the original model).

    Returns:
        [batch, heads, query length, head dim] attention output.
    """
    if block_mask is None:
        return F.scaled_dot_product_attention(query, key, value)
    return compiled_flex_attention(query, key, value, block_mask=block_mask)
