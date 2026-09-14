"""
Constants and helpers shared by the PixelDiT subpackage.

Trimmed copy of Mix_n_match/macros.py: only what pixeldit_model.py and text_encoder.py need is kept, and
WEIGHTS_DIR gained an environment override so an already downloaded weights folder can be reused as-is.
"""

import os
from pathlib import Path

# Folder of the tiled-diffusion project; weights and the prompt cache live under it.
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Environment variable that points the weights folder somewhere else, e.g. at a Mix_n_match checkout that already
# holds the PixelDiT checkpoint and the Gemma text encoder, so neither is downloaded a second time.
WEIGHTS_DIR_VARIABLE = "PIXELDIT_WEIGHTS_DIR"

# Where the PixelDiT checkpoint and the Gemma text encoder are downloaded once and loaded from afterwards.
WEIGHTS_DIR = Path(os.environ.get(WEIGHTS_DIR_VARIABLE) or (PROJECT_DIR / "weights")).expanduser()

# Encoded prompts, one file per prompt; the folder can be deleted at any time (prompts are then re-encoded).
PROMPT_CACHE_DIR = PROJECT_DIR / "prompt_cache"

# Side of one PixelDiT patch token in pixels. Image sizes and crop boundaries must be multiples of it.
PATCH_SIZE_PIXELS = 16


# Environment variable that turns the page cache dropping below off, for a machine that would rather keep it.
KEEP_PAGE_CACHE_VARIABLE = "MIX_N_MATCH_KEEP_PAGE_CACHE"

# Whether files are dropped from the page cache once they have been written or read. Reading the checkpoint fills
# the page cache, and WSL2 never hands that memory back to Windows, so the VM's footprint grows until the host
# starves; dropping a file's pages lets the next file reuse them instead.
# posix_fadvise exists on Linux (WSL included) but not on Windows or macOS.
DROP_PAGE_CACHE = hasattr(os, "posix_fadvise") and os.environ.get(KEEP_PAGE_CACHE_VARIABLE, "") != "1"


def drop_page_cache(path):
    """
    Drops the pages the kernel cached for a file; does nothing when DROP_PAGE_CACHE is off.

    The file itself is untouched: this only tells the kernel that its cached copy is no longer worth keeping, so
    reading the file again costs a disk read.

    Args:
        path: Path of the file whose cached pages are dropped.
    """
    if not DROP_PAGE_CACHE:
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)  # pages still dirty cannot be dropped
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(descriptor)
