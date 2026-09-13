"""Loading of the JSON configs produced by the saliency / cropping pipeline.

Only these keys are read:

    width, height          size of the final image
    num_crops              how many regions the final image is split into
    tiles_per_crop         how many candidate tiles we generate for each region
    crops                  [crop] -> {x, y, width, height}, the region each tile fills
    tile_prompts           [crop][candidate] -> positive prompt
    tile_negative_prompts  [crop][candidate] -> negative prompt

Every other key in the file (``saliency``, ``cfg_scale``, ``mode_filter``, ...) belongs
to the other project and is deliberately ignored, so configs can grow new fields
without anything here having to change.

``crops[i]`` describes the same crop as ``tile_prompts[i]`` -- the lists are parallel.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import List


@dataclass
class Crop:
    """A region of the final image, exactly as written in the config."""
    index: int          # position in the config's crops / tile_prompts lists
    x: int
    y: int
    width: int
    height: int

    @property
    def bottom(self):
        return self.y + self.height

    @property
    def right(self):
        return self.x + self.width


@dataclass
class CropConfig:
    image_width: int
    image_height: int
    num_crops: int
    tiles_per_crop: int
    crops: List[Crop]
    tile_prompts: List[List[str]]
    tile_negative_prompts: List[List[str]]
    config_name: str    # the config file's name, without its directory or .json suffix
    prefix: str         # the config's own 'prefix' field, or '' when it has none

    @property
    def output_name(self):
        """Name of the folder this run's images go into.

        The config's own 'prefix' is not enough on its own to tell two runs apart, so the
        config file's name comes first: 'example_portrait.json' with prefix 'cfg45' gives
        'example_portrait_cfg45'.
        """
        if not self.prefix or self.prefix == self.config_name:
            return self.config_name
        return f"{self.config_name}_{self.prefix}"

    def prompt(self, crop_idx, candidate_idx):
        return self.tile_prompts[crop_idx][candidate_idx]

    def negative_prompt(self, crop_idx, candidate_idx):
        return self.tile_negative_prompts[crop_idx][candidate_idx]

    @property
    def num_tiles(self):
        """Total number of tiles we are going to generate."""
        return self.num_crops * self.tiles_per_crop


def load_crop_config(path):
    """Read `path` and return a validated CropConfig."""
    with open(path, 'r') as f:
        raw = json.load(f)

    image_width = _require(raw, path, 'width')
    image_height = _require(raw, path, 'height')
    num_crops = _require(raw, path, 'num_crops')
    tiles_per_crop = _require(raw, path, 'tiles_per_crop')
    raw_crops = _require(raw, path, 'crops')
    tile_prompts = _require(raw, path, 'tile_prompts')
    tile_negative_prompts = _require(raw, path, 'tile_negative_prompts')

    # Both of these are only used to name the output folder. 'prefix' is optional.
    config_name = os.path.splitext(os.path.basename(path))[0]
    prefix = _clean_name(raw.get('prefix'))

    _validate_count(image_width, 'width', path)
    _validate_count(image_height, 'height', path)
    _validate_count(num_crops, 'num_crops', path)
    _validate_count(tiles_per_crop, 'tiles_per_crop', path)
    crops = _validate_crops(raw_crops, num_crops, image_width, image_height, path)
    _validate_prompt_table(tile_prompts, 'tile_prompts', num_crops, tiles_per_crop, path)
    _validate_prompt_table(tile_negative_prompts, 'tile_negative_prompts', num_crops, tiles_per_crop, path)

    return CropConfig(
        image_width=image_width,
        image_height=image_height,
        num_crops=num_crops,
        tiles_per_crop=tiles_per_crop,
        crops=crops,
        tile_prompts=tile_prompts,
        tile_negative_prompts=tile_negative_prompts,
        config_name=config_name,
        prefix=prefix,
    )


def _clean_name(value):
    """Keep a config's 'prefix' usable as part of a folder name."""
    if not isinstance(value, str):
        return ''
    return re.sub(r'[^A-Za-z0-9._-]+', '_', value).strip('_')


def _require(raw, path, key):
    if key not in raw:
        raise ValueError(f"Config '{path}' is missing the required key '{key}'.")
    return raw[key]


def _validate_count(value, key, path):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"Config '{path}': '{key}' must be an integer >= 1, got {value!r}.")


def _validate_crops(raw_crops, num_crops, image_width, image_height, path):
    """`crops` must hold one {x, y, width, height} box per crop, inside the image."""
    if not isinstance(raw_crops, list):
        raise ValueError(f"Config '{path}': 'crops' must be a list, got {type(raw_crops).__name__}.")

    if len(raw_crops) != num_crops:
        raise ValueError(
            f"Config '{path}': 'crops' has {len(raw_crops)} entries but 'num_crops' is {num_crops}. "
            f"There must be exactly one box per crop."
        )

    crops = []
    for crop_idx, box in enumerate(raw_crops):
        if not isinstance(box, dict):
            raise ValueError(f"Config '{path}': 'crops'[{crop_idx}] must be an object, "
                             f"got {type(box).__name__}.")
        for key in ('x', 'y', 'width', 'height'):
            if key not in box:
                raise ValueError(f"Config '{path}': 'crops'[{crop_idx}] is missing '{key}'.")
            if not isinstance(box[key], int) or isinstance(box[key], bool):
                raise ValueError(f"Config '{path}': 'crops'[{crop_idx}]['{key}'] must be an integer, "
                                 f"got {box[key]!r}.")

        crop = Crop(index=crop_idx, x=box['x'], y=box['y'], width=box['width'], height=box['height'])

        if crop.width < 1 or crop.height < 1:
            raise ValueError(f"Config '{path}': 'crops'[{crop_idx}] has a non-positive size "
                             f"({crop.width}x{crop.height}).")
        if crop.x < 0 or crop.y < 0 or crop.right > image_width or crop.bottom > image_height:
            raise ValueError(
                f"Config '{path}': 'crops'[{crop_idx}] ({crop.x},{crop.y} {crop.width}x{crop.height}) "
                f"does not fit inside the {image_width}x{image_height} image."
            )

        crops.append(crop)

    return crops


def _validate_prompt_table(table, key, num_crops, tiles_per_crop, path):
    """A prompt table is a list of `num_crops` lists, each holding `tiles_per_crop` strings."""
    if not isinstance(table, list):
        raise ValueError(f"Config '{path}': '{key}' must be a list of lists, got {type(table).__name__}.")

    if len(table) != num_crops:
        raise ValueError(
            f"Config '{path}': '{key}' has {len(table)} entries but 'num_crops' is {num_crops}. "
            f"There must be exactly one entry per crop."
        )

    for crop_idx, candidates in enumerate(table):
        if not isinstance(candidates, list):
            raise ValueError(
                f"Config '{path}': '{key}'[{crop_idx}] must be a list of prompts, "
                f"got {type(candidates).__name__}."
            )
        if len(candidates) != tiles_per_crop:
            raise ValueError(
                f"Config '{path}': '{key}'[{crop_idx}] holds {len(candidates)} prompts but "
                f"'tiles_per_crop' is {tiles_per_crop}. Every crop needs the same number of candidates."
            )
        for candidate_idx, prompt in enumerate(candidates):
            if not isinstance(prompt, str):
                raise ValueError(
                    f"Config '{path}': '{key}'[{crop_idx}][{candidate_idx}] must be a string, "
                    f"got {type(prompt).__name__}."
                )
