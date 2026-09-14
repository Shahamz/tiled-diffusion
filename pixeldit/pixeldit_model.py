"""
PixelDiT text-to-image transformer, copied from NVlabs/PixelDiT (commit 41f7300) and adapted for Mix_n_match.

Source files: pixdit_core/modules.py, pixdit_core/pixeldit_c2i.py, pixdit_core/pixeldit_t2i.py.
License: NVIDIA Source Code License (NSCLv1), see PixelDiT_LICENSE.txt.

Only the classes the text-to-image model needs for inference are kept; class names, attribute names and weight
shapes are unchanged, so the released checkpoint loads as-is. Removed: the ImageNet (class-to-image) model, REPA
tracking, the unused text-mask and precomputed-condition paths, zero-probability dropouts, and the PiT
post-modulation branch (the released checkpoint does not use it). Every changed spot is marked "Mix_n_match:".

The model has two stages:
    1. Patch stage: 16x16 pixel patches become tokens and go through MM-DiT blocks together with the text tokens.
    2. Pixel stage: PiT blocks refine every pixel, conditioned on its patch token; patches attend each other.

Mix_n_match changes: the forward takes a stack of S images (S = 1 before the split, k after) and runs them as ONE
token sequence. All S images reuse the same RoPE positions, so every tile sees itself at its crop's place in the
full image. Attention goes through attention.attention, masked by a TileAttentionLayout; with attend_on_avg the
crop-average keys/values are appended here. A layout may sort every image's patches by crop (patch_order), which is
applied once at the input and undone before the output; with a background crop it also keeps that crop's patches
only ONCE for all images (keep_background_once) and copies the result back into every image before the output. At
the saliency step the layout carries a saliency probe that reads the patch-stage queries and keys (saliency.py).
"""

import json
import math

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from .attention import attention, keep_background_once, keep_positions, restore_all_images
from .macros import WEIGHTS_DIR, drop_page_cache

# Hugging Face repository and files of the released 1.3B / 1024px text-to-image model.
CHECKPOINT_REPO_ID = "nvidia/PixelDiT-1300M-1024px"
CHECKPOINT_FILENAME = "pixeldit_t2i_v1.pth"
MODEL_CONFIG_FILENAME = "config.json"

# Keys of the repository's config.json that are constructor arguments of PixDiT_T2I.
ARCHITECTURE_KEYS = (
    "in_channels", "patch_size", "num_groups", "hidden_size", "pixel_hidden_size", "pixel_attn_hidden_size",
    "pixel_num_groups", "patch_depth", "pixel_depth", "txt_embed_dim", "txt_max_length", "use_text_rope",
    "text_rope_theta", "use_pixel_abs_pos",
)

# Checkpoint key prefix added by PixelDiT's training wrapper, and the training-only REPA projector to drop.
CHECKPOINT_WRAPPER_PREFIX = "core."
CHECKPOINT_TRAINING_ONLY_PREFIX = "_repa_projector."


# ----------------------------------------------------------------------------------------------------------------
# Helpers (from modules.py)
# ----------------------------------------------------------------------------------------------------------------


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    Absolute 2D sine-cosine position embedding of a grid.

    Args:
        embed_dim: embedding size (even).
        grid: [2, 1, height, width] array of (x, y) coordinates.

    Returns:
        [height * width, embed_dim] numpy array.
    """
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Absolute 1D sine-cosine position embedding.

    Args:
        embed_dim: embedding size per position (even).
        pos: array of positions, flattened to [M].

    Returns:
        [M, embed_dim] numpy array.
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def apply_adaln(x, shift, scale):
    """
    Adaptive layer-norm modulation.

    Args:
        x: normalized tensor.
        shift, scale: modulation tensors broadcastable to x.

    Returns:
        x * (1 + scale) + shift.
    """
    return x * (1 + scale) + shift


def precompute_freqs_cis_2d(dim, height, width, theta=10000.0, scale=16.0):
    """
    2D rotary position frequencies of a height x width token grid. Positions are spread over [0, scale] on both
    axes regardless of the grid size.

    Args:
        dim: attention head dimension.
        height, width: token grid size.
        theta: RoPE base.
        scale: coordinate range of the grid.

    Returns:
        [height * width, dim // 2] complex tensor.
    """
    x_pos = torch.linspace(0, scale, width)
    y_pos = torch.linspace(0, scale, height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    x_freqs = torch.outer(x_pos, freqs).float()
    y_freqs = torch.outer(y_pos, freqs).float()
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1)
    return freqs_cis.reshape(height * width, -1)


def apply_rotary_emb(xq, xk, freqs_cis):
    """
    Rotates queries and keys by their positions (RoPE).

    Args:
        xq, xk: [batch, length, heads, head dim] queries and keys.
        freqs_cis: [length, head dim // 2] complex frequencies.

    Returns:
        (rotated queries, rotated keys) with the input shapes and dtypes.
    """
    freqs_cis = freqs_cis[None, :, None, :]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def average_over_tiles(tokens, layout):
    """
    Mix_n_match: averages the tile tokens over the k tiles. Image s holds tile s of every crop, so the result holds,
    at every tile patch location, the average of the k tiles of the crop covering it. The background crop's
    patches, which follow the tile tokens and exist once, are left out.

    Args:
        tokens: [batch, k * tile patches (+ background patches), ...] tensor (tile-major order).
        layout: TileAttentionLayout.

    Returns:
        [batch, tile patches, ...] tensor.
    """
    tile_tokens = tokens[:, : layout.num_tiles * layout.tile_patch_count]
    return tile_tokens.view(tile_tokens.shape[0], layout.num_tiles, layout.tile_patch_count,
                            *tokens.shape[2:]).mean(dim=1)


def crop_average_keys_values(hidden_states, keys, values, qkv_layer, key_norm, positions, num_heads, layout):
    """
    Mix_n_match: builds the crop-average keys and values that tiles attend with attend_on_avg.

    Args:
        hidden_states: [batch, image tokens, dim] normalized attention input of the image tokens.
        keys: [batch, image tokens, heads, head dim] image keys after norm and RoPE.
        values: [batch, image tokens, heads, head dim] image values.
        qkv_layer: the attention's image qkv projection.
        key_norm: the attention's image key norm.
        positions: [image tokens, head dim // 2] image RoPE frequencies.
        num_heads: number of attention heads.
        layout: TileAttentionLayout; average_target picks the mode:
            "keys_values"   - average the keys and values themselves;
            "hidden_states" - average the attention input, then project, normalize and rotate it into a key/value.

    Returns:
        (average keys, average values), each [batch, patches, heads, head dim].
    """
    if layout.average_target == "keys_values":
        return average_over_tiles(keys, layout), average_over_tiles(values, layout)

    average_hidden = average_over_tiles(hidden_states, layout)
    batch, num_patches, dim = average_hidden.shape
    qkv = qkv_layer(average_hidden).reshape(batch, num_patches, 3, num_heads, dim // num_heads)
    average_keys = key_norm(qkv[:, :, 1])
    # All tiles share the same positions, so the first num_patches positions are the positions of the average.
    _, average_keys = apply_rotary_emb(average_keys, average_keys, freqs_cis=positions[:num_patches])
    return average_keys, qkv[:, :, 2]


# ----------------------------------------------------------------------------------------------------------------
# Layers (from modules.py and pixeldit_c2i.py)
# ----------------------------------------------------------------------------------------------------------------


class TimestepConditioner(nn.Module):
    """Embeds the diffusion timestep into the model's hidden size."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        """
        Args:
            hidden_size: output embedding size.
            frequency_embedding_size: size of the sinusoidal timestep features.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10):
        """
        Sinusoidal timestep features.

        Args:
            t: [N] timesteps.
            dim: feature size.
            max_period: controls the lowest frequency.

        Returns:
            [N, dim] features.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
        args = t[..., None].float() * freqs[None, ...]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        """
        Args:
            t: [N] timesteps.

        Returns:
            [N, hidden_size] timestep embeddings.
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(next(self.mlp.parameters()).dtype))


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned scale."""

    def __init__(self, hidden_size, eps=1e-6):
        """
        Args:
            hidden_size: size of the normalized (last) dimension.
            eps: numerical stability constant.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [..., hidden_size] tensor.

        Returns:
            Normalized and scaled tensor with the input dtype.
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class FeedForward(nn.Module):
    """SwiGLU feed-forward block used by the patch stage."""

    def __init__(self, dim, hidden_dim):
        """
        Args:
            dim: input and output size.
            hidden_dim: nominal hidden size (2/3 of it is used, as in SwiGLU).
        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        """
        Args:
            x: [..., dim] tensor.

        Returns:
            [..., dim] tensor.
        """
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class MLP(nn.Module):
    """GELU MLP used by the pixel stage."""

    def __init__(self, dim, mlp_ratio=4.0):
        """
        Args:
            dim: input and output size.
            mlp_ratio: hidden size as a multiple of dim.
        """
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        """
        Args:
            x: [..., dim] tensor.

        Returns:
            [..., dim] tensor.
        """
        return self.fc2(self.act(self.fc1(x)))


class FinalLayer(nn.Module):
    """Projects pixel features back to RGB."""

    def __init__(self, hidden_size, out_channels):
        """
        Args:
            hidden_size: pixel feature size.
            out_channels: number of output channels (3).
        """
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x):
        """
        Args:
            x: [..., hidden_size] tensor.

        Returns:
            [..., out_channels] tensor.
        """
        return self.linear(self.norm(x))


class PatchTokenEmbedder(nn.Module):
    """Linear embedding (plus optional norm) of flattened patches or text features."""

    def __init__(self, in_chans=3, embed_dim=768, norm_layer=None, bias=True):
        """
        Args:
            in_chans: input feature size.
            embed_dim: output size.
            norm_layer: optional normalization class applied after the projection.
            bias: whether the projection has a bias.
        """
        super().__init__()
        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: [..., in_chans] tensor.

        Returns:
            [..., embed_dim] tensor.
        """
        return self.norm(self.proj(x))


class PixelTokenEmbedder(nn.Module):
    """Embeds every pixel and groups the pixels by patch for the pixel stage."""

    def __init__(self, in_channels, hidden_size_output, use_pixel_abs_pos=True):
        """
        Args:
            in_channels: image channels (3).
            hidden_size_output: pixel feature size.
            use_pixel_abs_pos: whether to add an absolute sine-cosine position embedding per pixel.
        """
        super().__init__()
        self.hidden_size_output = int(hidden_size_output)
        self.use_pixel_abs_pos = bool(use_pixel_abs_pos)
        self.proj = nn.Linear(int(in_channels), self.hidden_size_output, bias=True)
        self._pos_cache = dict()

    def _fetch_pixel_pos_image(self, height, width, device, dtype):
        """
        Absolute per-pixel position embedding of a height x width image (cached).
        Mix_n_match: the original's separate square-image branch computed the same values and was removed.

        Returns:
            [height * width, hidden_size_output] tensor.
        """
        key = ("image", height, width)
        if key not in self._pos_cache:
            grid = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
            grid = np.stack(grid, axis=0).reshape(2, 1, height, width)
            self._pos_cache[key] = torch.from_numpy(get_2d_sincos_pos_embed_from_grid(self.hidden_size_output, grid))
        return self._pos_cache[key].to(device=device, dtype=dtype)

    def forward(self, inputs, img_height, img_width, patch_size):
        """
        Args:
            inputs: [N, channels, height, width] images.
            img_height, img_width: image size in pixels.
            patch_size: patch side in pixels.

        Returns:
            [N * patches, patch_size ** 2, hidden_size_output] pixel features, grouped by patch.
        """
        B, C, H, W = inputs.shape
        Hs, Ws = H // patch_size, W // patch_size
        x = self.proj(inputs.permute(0, 2, 3, 1).contiguous())
        if self.use_pixel_abs_pos:
            pos_full = self._fetch_pixel_pos_image(H, W, inputs.device, inputs.dtype)
            x = x + pos_full.view(H, W, self.hidden_size_output).unsqueeze(0)
        x = x.view(B, Hs, patch_size, Ws, patch_size, self.hidden_size_output)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(B * Hs * Ws, patch_size * patch_size, self.hidden_size_output)


class RotaryAttention(nn.Module):
    """Self-attention with RoPE and query/key RMS norms, used between patches in the pixel stage."""

    def __init__(self, dim, num_heads=8, qkv_bias=False):
        """
        Args:
            dim: token size.
            num_heads: number of attention heads.
            qkv_bias: whether the qkv projection has a bias.
        """
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, pos, layout=None):
        """
        Args:
            x: [batch, k * patches, dim] tokens.
            pos: [k * patches, head dim // 2] RoPE frequencies.
            layout: TileAttentionLayout, or None for full attention (before the split).

        Returns:
            [batch, k * patches, dim] tensor.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        q, k, v = self.q_norm(qkv[0]), self.k_norm(qkv[1]), qkv[2]
        q, k = apply_rotary_emb(q, k, freqs_cis=pos)
        # Mix_n_match: append the crop-average keys/values and use the pixel-stage tile mask.
        if layout is not None and layout.average_target is not None:
            average_k, average_v = crop_average_keys_values(x, k, v, self.qkv, self.k_norm, pos, self.num_heads, layout)
            k, v = torch.cat([k, average_k], dim=1), torch.cat([v, average_v], dim=1)
        block_mask = layout.pixel_block_mask if layout is not None else None
        x = attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class PiTBlock(nn.Module):
    """Pixel-stage block: per-pixel modulation from the patch token, and attention between compressed patches."""

    def __init__(self, pixel_hidden_size, patch_hidden_size, patch_size, num_heads, mlp_ratio=4.0,
                 attn_hidden_size=None, attn_num_heads=None, rope_fn=None):
        """
        Args:
            pixel_hidden_size: pixel feature size.
            patch_hidden_size: patch token (condition) size.
            patch_size: patch side in pixels.
            num_heads: default number of attention heads.
            mlp_ratio: MLP hidden size as a multiple of pixel_hidden_size.
            attn_hidden_size: size of a compressed patch in attention (defaults to patch_hidden_size).
            attn_num_heads: attention heads (defaults to num_heads).
            rope_fn: function building the RoPE frequencies (defaults to precompute_freqs_cis_2d).
        """
        super().__init__()
        self.pixel_dim = int(pixel_hidden_size)
        self.attn_dim = int(attn_hidden_size) if attn_hidden_size is not None else int(patch_hidden_size)
        self.num_heads = int(attn_num_heads) if attn_num_heads is not None else int(num_heads)
        p2 = int(patch_size) ** 2
        self.compress_to_attn = nn.Linear(p2 * self.pixel_dim, self.attn_dim, bias=True)
        self.expand_from_attn = nn.Linear(self.attn_dim, p2 * self.pixel_dim, bias=True)
        self.norm1 = RMSNorm(self.pixel_dim, eps=1e-6)
        self.attn = RotaryAttention(self.attn_dim, num_heads=self.num_heads, qkv_bias=False)
        self.norm2 = RMSNorm(self.pixel_dim, eps=1e-6)
        self.mlp = MLP(self.pixel_dim, mlp_ratio=mlp_ratio)
        self.adaLN_modulation = nn.Sequential(nn.Linear(int(patch_hidden_size), 6 * self.pixel_dim * p2, bias=True))
        self._pos_cache = dict()
        self._rope_fn = rope_fn if rope_fn is not None else precompute_freqs_cis_2d

    def _fetch_pos(self, height, width, device):
        """
        RoPE frequencies of a height x width patch grid (cached).

        Returns:
            [height * width, head dim // 2] complex tensor.
        """
        key = (height, width)
        if key not in self._pos_cache:
            self._pos_cache[key] = self._rope_fn(self.attn_dim // self.num_heads, height, width)
        return self._pos_cache[key].to(device)

    def forward(self, x, s_cond, image_height, image_width, patch_size, num_images=1, layout=None):
        """
        Args:
            x: [batch * sequence length, patch_size ** 2, pixel dim] pixel features.
            s_cond: [batch * sequence length, patch hidden size] patch tokens used as condition.
            image_height, image_width: image size in pixels.
            patch_size: patch side in pixels.
            num_images: Mix_n_match: number of images (k) joined into one sequence.
            layout: Mix_n_match: TileAttentionLayout, or None for full attention.

        Returns:
            Updated pixel features with the shape of x.
        """
        BL, P2, C = x.shape
        Hs, Ws = image_height // patch_size, image_width // patch_size
        # Mix_n_match: all images form one sequence, with the background crop's patches kept once (attention.py).
        pos_comp = self._fetch_pos(Hs, Ws, x.device)
        if layout is None:
            sequence_length = num_images * Hs * Ws
            pos_comp = pos_comp.repeat(num_images, 1)
        else:
            if layout.patch_order is not None:
                pos_comp = pos_comp[layout.patch_order]
            pos_comp = keep_positions(pos_comp, layout, num_images)
            sequence_length = pos_comp.shape[0]
        B = BL // sequence_length
        cond_params = self.adaLN_modulation(s_cond).view(BL, P2, 6 * self.pixel_dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(cond_params, 6, dim=-1)
        x_norm = apply_adaln(self.norm1(x), shift_msa, scale_msa)
        x_comp = self.compress_to_attn(x_norm.view(BL, P2 * self.pixel_dim)).view(B, sequence_length, self.attn_dim)
        attn_out = self.attn(x_comp, pos_comp, layout)
        attn_exp = self.expand_from_attn(attn_out.view(BL, self.attn_dim)).view(BL, P2, self.pixel_dim)
        x = x + gate_msa * attn_exp
        return x + gate_mlp * self.mlp(apply_adaln(self.norm2(x), shift_mlp, scale_mlp))


# ----------------------------------------------------------------------------------------------------------------
# Patch stage and full model (from pixeldit_t2i.py)
# ----------------------------------------------------------------------------------------------------------------


class MMDiTJointAttention(nn.Module):
    """Joint attention over text and image tokens with separate projections per modality."""

    def __init__(self, dim, num_heads=8, qkv_bias=False):
        """
        Args:
            dim: token size.
            num_heads: number of attention heads.
            qkv_bias: whether the qkv projections have a bias.
        """
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv_x = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_y = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm_x = RMSNorm(self.head_dim)
        self.k_norm_x = RMSNorm(self.head_dim)
        self.q_norm_y = RMSNorm(self.head_dim)
        self.k_norm_y = RMSNorm(self.head_dim)
        self.proj_x = nn.Linear(dim, dim)
        self.proj_y = nn.Linear(dim, dim)

    def forward(self, x, y, pos_img, pos_txt=None, layout=None):
        """
        Args:
            x: [batch, k * patches, dim] image tokens.
            y: [batch, prompts * prompt length, dim] text tokens.
            pos_img: [k * patches, head dim // 2] image RoPE frequencies.
            pos_txt: [prompts * prompt length, head dim // 2] text RoPE frequencies, or None.
            layout: Mix_n_match: TileAttentionLayout, or None for full attention (before the split).

        Returns:
            (image output, text output) with the shapes of x and y.
        """
        B, Nx, C = x.shape
        Ny = y.shape[1]
        qkv_x = self.qkv_x(x).reshape(B, Nx, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        qx, kx, vx = self.q_norm_x(qkv_x[0]), self.k_norm_x(qkv_x[1]), qkv_x[2]
        qkv_y = self.qkv_y(y).reshape(B, Ny, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        qy, ky, vy = self.q_norm_y(qkv_y[0]), self.k_norm_y(qkv_y[1]), qkv_y[2]

        qx, kx = apply_rotary_emb(qx, kx, freqs_cis=pos_img)
        if pos_txt is not None:
            qy, ky = apply_rotary_emb(qy, ky, freqs_cis=pos_txt)
        # Mix_n_match: at the saliency step, prompt-image attention scores are measured on the side.
        if layout is not None and layout.saliency_probe is not None:
            layout.saliency_probe.record(qx, kx, qy, ky)

        # Token order is [text, image]; Mix_n_match: crop-average keys/values go last, and the tile mask is used.
        q_joint = torch.cat([qy, qx], dim=1)
        k_joint = torch.cat([ky, kx], dim=1)
        v_joint = torch.cat([vy, vx], dim=1)
        if layout is not None and layout.average_target is not None:
            average_k, average_v = crop_average_keys_values(
                x, kx, vx, self.qkv_x, self.k_norm_x, pos_img, self.num_heads, layout
            )
            k_joint, v_joint = torch.cat([k_joint, average_k], dim=1), torch.cat([v_joint, average_v], dim=1)
        block_mask = layout.joint_block_mask if layout is not None else None
        out_joint = attention(q_joint.transpose(1, 2), k_joint.transpose(1, 2), v_joint.transpose(1, 2), block_mask)

        out_joint = out_joint.transpose(1, 2)
        out_y = out_joint[:, :Ny].reshape(B, Ny, C)
        out_x = out_joint[:, Ny:].reshape(B, Nx, C)
        return self.proj_x(out_x), self.proj_y(out_y)


class MMDiTBlockT2I(nn.Module):
    """Patch-stage MM-DiT block: joint attention and per-modality MLPs, modulated by the timestep."""

    def __init__(self, hidden_size, groups, mlp_ratio=4.0):
        """
        Args:
            hidden_size: token size.
            groups: number of attention heads.
            mlp_ratio: MLP hidden size as a multiple of hidden_size.
        """
        super().__init__()
        self.norm_x1 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_y1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = MMDiTJointAttention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm_x2 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_y2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_x = FeedForward(hidden_size, mlp_hidden_dim)
        self.mlp_y = FeedForward(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation_img = nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.adaLN_modulation_txt = nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, y, c, pos_img, pos_txt=None, layout=None):
        """
        Args:
            x: [batch, k * patches, hidden] image tokens.
            y: [batch, text length, hidden] text tokens.
            c: [batch, 1, hidden] timestep condition.
            pos_img, pos_txt: RoPE frequencies of the image and text tokens.
            layout: Mix_n_match: TileAttentionLayout, or None for full attention.

        Returns:
            (updated x, updated y).
        """
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = self.adaLN_modulation_img(c).chunk(6, dim=-1)
        shift_msa_y, scale_msa_y, gate_msa_y, shift_mlp_y, scale_mlp_y, gate_mlp_y = self.adaLN_modulation_txt(c).chunk(6, dim=-1)

        x_norm = apply_adaln(self.norm_x1(x), shift_msa_x, scale_msa_x)
        y_norm = apply_adaln(self.norm_y1(y), shift_msa_y, scale_msa_y)
        attn_x, attn_y = self.attn(x_norm, y_norm, pos_img, pos_txt, layout)
        x = x + gate_msa_x * attn_x
        y = y + gate_msa_y * attn_y

        x = x + gate_mlp_x * self.mlp_x(apply_adaln(self.norm_x2(x), shift_mlp_x, scale_mlp_x))
        y = y + gate_mlp_y * self.mlp_y(apply_adaln(self.norm_y2(y), shift_mlp_y, scale_mlp_y))
        return x, y


class PixDiT_T2I(nn.Module):
    """PixelDiT text-to-image model: predicts the flow velocity (noise - image) of pixel-space images."""

    def __init__(self, in_channels=3, num_groups=16, hidden_size=1152, pixel_hidden_size=64,
                 pixel_attn_hidden_size=None, pixel_num_groups=None, patch_depth=26, pixel_depth=2, patch_size=16,
                 txt_embed_dim=4096, txt_max_length=1024, use_text_rope=True, text_rope_theta=10000.0,
                 use_pixel_abs_pos=True):
        """
        Args:
            in_channels: image channels (3).
            num_groups: attention heads of the patch stage.
            hidden_size: patch token size.
            pixel_hidden_size: pixel feature size.
            pixel_attn_hidden_size: compressed patch size in pixel-stage attention.
            pixel_num_groups: attention heads of the pixel stage.
            patch_depth: number of MM-DiT blocks.
            pixel_depth: number of PiT blocks.
            patch_size: patch side in pixels.
            txt_embed_dim: text encoder feature size.
            txt_max_length: maximum tokens per prompt.
            use_text_rope: whether text tokens get 1D RoPE.
            text_rope_theta: text RoPE base.
            use_pixel_abs_pos: whether pixels get an absolute position embedding.
        """
        super().__init__()
        self.out_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.num_groups = int(num_groups)
        self.patch_size = int(patch_size)
        self.txt_max_length = int(txt_max_length)
        self.use_text_rope = bool(use_text_rope)
        self.text_rope_theta = float(text_rope_theta)

        self.pixel_embedder = PixelTokenEmbedder(in_channels, pixel_hidden_size, use_pixel_abs_pos=use_pixel_abs_pos)
        self.s_embedder = PatchTokenEmbedder(in_channels * self.patch_size**2, self.hidden_size, bias=True)
        self.t_embedder = TimestepConditioner(self.hidden_size)
        self.y_embedder = PatchTokenEmbedder(int(txt_embed_dim), self.hidden_size, bias=True, norm_layer=RMSNorm)
        self.y_pos_embedding = nn.Parameter(torch.randn(1, self.txt_max_length, self.hidden_size))
        self.patch_blocks = nn.ModuleList([MMDiTBlockT2I(self.hidden_size, self.num_groups) for _ in range(int(patch_depth))])
        pixel_attn_hidden_size = int(pixel_attn_hidden_size) if pixel_attn_hidden_size is not None else self.hidden_size
        pixel_num_groups = int(pixel_num_groups) if pixel_num_groups is not None else self.num_groups
        self.pixel_blocks = nn.ModuleList([
            PiTBlock(int(pixel_hidden_size), self.hidden_size, patch_size=self.patch_size, num_heads=self.num_groups,
                     mlp_ratio=4.0, attn_hidden_size=pixel_attn_hidden_size, attn_num_heads=pixel_num_groups,
                     rope_fn=precompute_freqs_cis_2d)
            for _ in range(int(pixel_depth))
        ])
        self.final_layer = FinalLayer(int(pixel_hidden_size), self.out_channels)
        self.precompute_pos = dict()
        self.precompute_pos_txt = dict()

    @property
    def device(self):
        """Mix_n_match: device of the weights (DiffusionPipeline reads it from its components)."""
        return next(self.parameters()).device

    def fetch_pos(self, height, width, device):
        """
        Patch-stage RoPE frequencies of a height x width patch grid (cached).

        Returns:
            [height * width, head dim // 2] complex tensor.
        """
        if (height, width) not in self.precompute_pos:
            self.precompute_pos[(height, width)] = precompute_freqs_cis_2d(self.hidden_size // self.num_groups, height, width)
        return self.precompute_pos[(height, width)].to(device)

    def fetch_pos_text(self, length, device):
        """
        1D RoPE frequencies of a prompt of the given length (cached).

        Returns:
            [length, head dim // 2] complex tensor.
        """
        if length not in self.precompute_pos_txt:
            head_dim = self.hidden_size // self.num_groups
            freqs = 1.0 / (self.text_rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
            angles = torch.arange(0, length).float().unsqueeze(1) * freqs.unsqueeze(0)
            self.precompute_pos_txt[length] = torch.polar(torch.ones_like(angles), angles)
        return self.precompute_pos_txt[length].to(device)

    def forward(self, images, timestep, text_embeds, layout=None):
        """
        Mix_n_match: runs a stack of S images and P prompts as ONE sequence per batch element.

        Args:
            images: [batch, S, 3, H, W] noisy images in [-1, 1] scale; S = 1 before the split, k after (image s
                holds tile s of every crop).
            timestep: [batch] timesteps in [0, 1000].
            text_embeds: [batch, P, prompt length, text dim] prompt features; prompt 0 is the background prompt,
                prompt 1 + crop * k + tile is that tile's prompt.
            layout: TileAttentionLayout after the split or at the saliency step, None otherwise (the original,
                unmasked model).

        Returns:
            [batch, S, 3, H, W] predicted flow velocity.
        """
        B, S, _, H, W = images.shape
        P = text_embeds.shape[1]
        Hs, Ws = H // self.patch_size, W // self.patch_size
        L = Hs * Ws
        patch_order = layout.patch_order if layout is not None else None

        # Mix_n_match: patchify every image and join them into one sequence; all images reuse the same positions,
        # so sibling tiles share their crop's RoPE. With a patch order (attention.py), every image's patches and
        # their positions are sorted by crop here and put back before the final fold.
        flat_images = images.reshape(B * S, *images.shape[2:])
        x_patches = torch.nn.functional.unfold(flat_images, kernel_size=self.patch_size, stride=self.patch_size)
        x_patches = x_patches.transpose(1, 2)
        pos = self.fetch_pos(Hs, Ws, images.device)
        if patch_order is not None:
            x_patches, pos = x_patches[:, patch_order], pos[patch_order]
        if layout is None:
            x_patches, pos = x_patches.reshape(B, S * L, -1), pos.repeat(S, 1)
        else:
            # Mix_n_match: the background crop's patches are kept once, from image 0 (attention.py).
            x_patches = keep_background_once(x_patches, layout, S)
            pos = keep_positions(pos, layout, S)

        t_emb = self.t_embedder(timestep.view(-1)).view(B, -1, self.hidden_size)
        condition = torch.nn.functional.silu(t_emb)

        # Mix_n_match: every prompt gets its own learned positions and text RoPE, as if it were alone.
        text_length = min(text_embeds.shape[2], self.txt_max_length)
        y_emb = self.y_embedder(text_embeds[:, :, :text_length]) + self.y_pos_embedding[:, :text_length].to(images.dtype)
        y_emb = y_emb.reshape(B, P * text_length, self.hidden_size)
        pos_txt = self.fetch_pos_text(text_length, images.device).repeat(P, 1) if self.use_text_rope else None

        s = self.s_embedder(x_patches)
        for block in self.patch_blocks:
            s, y_emb = block(s, y_emb, condition, pos, pos_txt, layout)
        s = torch.nn.functional.silu(t_emb + s)

        sequence_length = s.shape[1]
        s_cond = s.reshape(B * sequence_length, self.hidden_size)
        P2 = self.patch_size * self.patch_size
        x_pixels = self.pixel_embedder(flat_images, img_height=H, img_width=W, patch_size=self.patch_size)
        x_pixels = x_pixels.view(B * S, L, P2, -1)
        if patch_order is not None:
            x_pixels = x_pixels[:, patch_order]
        if layout is not None:
            x_pixels = keep_background_once(x_pixels, layout, S)
        x_pixels = x_pixels.reshape(B * sequence_length, P2, -1)
        for block in self.pixel_blocks:
            x_pixels = block(x_pixels, s_cond, H, W, self.patch_size, num_images=S, layout=layout)

        # Mix_n_match: every image gets its tile patches back and a copy of the single background region.
        x_pixels = self.final_layer(x_pixels).view(B, sequence_length, P2, self.out_channels)
        x_pixels = (restore_all_images(x_pixels, layout, S) if layout is not None
                    else x_pixels.view(B * S, L, P2, self.out_channels))
        if patch_order is not None:
            x_pixels = x_pixels[:, layout.patch_restore]
        x_pixels = x_pixels.permute(0, 3, 2, 1).reshape(B * S, self.out_channels * P2, L)
        x_img = torch.nn.functional.fold(x_pixels, (H, W), kernel_size=self.patch_size, stride=self.patch_size)
        return x_img.view(B, S, self.out_channels, H, W)


def read_checkpoint(path):
    """
    Reads the checkpoint with its tensors memory-mapped, so they are not copied into host RAM.

    Memory mapping needs the zipfile format torch.save has written since 1.6; an older checkpoint is read the
    ordinary way, which costs its full size in host RAM.

    Args:
        path: Path of the checkpoint file.

    Returns:
        The checkpoint dict.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, ValueError) as error:
        print(f"Note: {path.name} cannot be memory-mapped ({error}); reading it into RAM instead")
        return torch.load(path, map_location="cpu", weights_only=False)


def load_pixeldit(device, dtype):
    """
    Builds PixDiT_T2I from the released checkpoint, downloading it into WEIGHTS_DIR on first use.

    The weights are never materialized in host RAM: the model is built on the meta device, whose tensors have a
    shape but no storage, and the memory-mapped checkpoint becomes its parameters directly. Building the model
    the ordinary way would hold a randomly initialized copy and the checkpoint at once, about 10 GB for this
    1300M-parameter checkpoint, all of it before anything reaches the GPU.

    Args:
        device: torch device to place the model on.
        dtype: model dtype (bfloat16 in the original inference).

    Returns:
        (model in eval mode, repository config dict; its "scheduler" entry holds the flow shift).

    Raises:
        RuntimeError: if the checkpoint weights do not match the model exactly.
    """
    for filename in (MODEL_CONFIG_FILENAME, CHECKPOINT_FILENAME):
        if not (WEIGHTS_DIR / filename).exists():
            print(f"Downloading {CHECKPOINT_REPO_ID}/{filename} to {WEIGHTS_DIR} (one time only)...")
            hf_hub_download(CHECKPOINT_REPO_ID, filename, local_dir=WEIGHTS_DIR)
    model_config = json.loads((WEIGHTS_DIR / MODEL_CONFIG_FILENAME).read_text())
    with torch.device("meta"):
        model = PixDiT_T2I(**{key: model_config[key] for key in ARCHITECTURE_KEYS})

    checkpoint_path = WEIGHTS_DIR / CHECKPOINT_FILENAME
    checkpoint = read_checkpoint(checkpoint_path)
    state_dict = {
        key.removeprefix(CHECKPOINT_WRAPPER_PREFIX): value
        for key, value in checkpoint["state_dict"].items()
        if not key.startswith(CHECKPOINT_TRAINING_ONLY_PREFIX)
    }
    # A meta tensor has no storage to copy into, so every parameter takes the checkpoint's tensor itself; strict
    # loading is what guarantees none is left behind on the meta device.
    model.load_state_dict(state_dict, strict=True, assign=True)
    model = model.to(device=device, dtype=dtype).eval()
    # The weights are on the GPU now, so the mapped file is no longer needed, cached or otherwise.
    del state_dict, checkpoint
    drop_page_cache(checkpoint_path)
    return model, model_config
