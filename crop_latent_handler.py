"""A corrected `LatentHandler.tile` for tiles that do not all have the same size.

Why this file exists
--------------------
`LatentHandler.tile` (latent_handler.py:22-30) reads the height and width of the *source*
tile and then uses them to slice the *target* tile:

    B, F, H, W = source_latent.pre_latent.size()
    ...
    else:  # Down
        tensor = target_latent.pre_latent[:, :, H - 2 * max_width: H - max_width, :]

Every tile in the original project is 512x512, so source and target dimensions always
agree and this is harmless. Once crops have their own sizes it is not: the band is read
from the wrong rows of the target, and when the source is taller than the target the
slice comes back short and the assignment raises a shape error.

The other two handler methods are fine as they are. In this layout a connection group
only ever contains candidates of a single crop (all of a crop's candidates share their
side descriptors), so `apply_similarity_constraint` and `apply_random_padding_constraint`
never copy between tiles of different sizes.

The fix below is the same code with the target's own dimensions used for the target's
slice. It is installed by overwriting the name `model` imported, so no existing file in
the repository has to change.
"""

import torch

from config import TILING_ROTATION_MATRIX
from latent_handler import LatentHandler


class CropLatentHandler(LatentHandler):

    @staticmethod
    def tile(latents_arr, step, groups, max_width=10):
        for key, val in groups.items():
            target_latent_idx = val['target_latent_idx']
            target_side_idx = val['target_side_idx']
            len_target = len(target_latent_idx)
            latent_source_id, side_source_id = map(int, key.split('_'))
            candidate_idx = step % len_target
            chosen_latent_idx = target_latent_idx[candidate_idx]
            chosen_side_idx = target_side_idx[candidate_idx]
            source_latent = latents_arr[latent_source_id]
            target_latent = latents_arr[chosen_latent_idx]
            B, F, H, W = source_latent.pre_latent.size()
            # The only change from LatentHandler.tile: the target is sliced using its own size.
            _, _, target_H, target_W = target_latent.pre_latent.size()
            if chosen_side_idx == 0:  # Right
                tensor = target_latent.pre_latent[:, :, :, target_W - 2 * max_width: target_W - max_width]
            elif chosen_side_idx == 1:  # Left
                tensor = target_latent.pre_latent[:, :, :, max_width: 2 * max_width]
            elif chosen_side_idx == 2:  # Up
                tensor = target_latent.pre_latent[:, :, max_width: 2 * max_width, :]
            else:  # Down
                tensor = target_latent.pre_latent[:, :, target_H - 2 * max_width: target_H - max_width, :]
            rotation_value = TILING_ROTATION_MATRIX[side_source_id][chosen_side_idx]
            rotated_tensor = torch.rot90(tensor, rotation_value, [2, 3])

            # Writing into the source, so the source's own H and W are the right ones here.
            if side_source_id == 0:  # Right
                source_latent.post_latent[:, :, :, W - max_width:] = rotated_tensor
            elif side_source_id == 1:  # Left
                source_latent.post_latent[:, :, :, :max_width] = rotated_tensor
            elif side_source_id == 2:  # Up
                source_latent.post_latent[:, :, :max_width, :] = rotated_tensor
            else:  # Down
                source_latent.post_latent[:, :, H - max_width:, :] = rotated_tensor

        return latents_arr


def install():
    """Make model.py use CropLatentHandler instead of LatentHandler.

    model.py does `from latent_handler import LatentHandler`, so the name to replace is
    the one in the model module's namespace. Call this after importing model and before
    running it.
    """
    import model
    model.LatentHandler = CropLatentHandler
