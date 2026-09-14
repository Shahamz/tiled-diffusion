"""
Gemma prompt encoding for the PixelDiT path, lifted from Mix_n_match/pipeline_mix_n_match.py.

PixelDiT is conditioned on Gemma-2-2b-it features, not on CLIP ones, so a LatentClass cannot build its own
embeddings the way LatentClass.set_text_embs does for Stable Diffusion. Gemma (about 5 GB in bfloat16) and the
transformer (about 2.6 GB) also do not fit on an 8 GB GPU together, so encoding is its own phase: set_text_embeddings
encodes every tile's prompts, frees Gemma, and only then may the caller load the transformer.

Encoded prompts are cached on disk (PROMPT_CACHE_DIR), so Gemma is loaded only when some prompt was never encoded
before. The cache key and the entry format are the ones Mix_n_match uses, so its prompt_cache folder can be
symlinked here and reused.
"""

import gc
import hashlib

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, Gemma2Model

from .macros import PROMPT_CACHE_DIR, WEIGHTS_DIR

# The Gemma-2-2b-it copy PixelDiT was trained with, its local folder, and the files needed from it.
TEXT_ENCODER_REPO_ID = "Efficient-Large-Model/gemma-2-2b-it"
TEXT_ENCODER_DIR = WEIGHTS_DIR / "gemma-2-2b-it"
TEXT_ENCODER_FILE_PATTERNS = ["*.json", "model-*.safetensors", "tokenizer.model"]

# Prompts encoded by Gemma at once.
ENCODE_BATCH_SIZE = 16

# Tokens per prompt fed to the transformer (PixelDiT's model_max_length).
PROMPT_LENGTH = 300

# dtype of both models (PixelDiT's inference runs in bfloat16).
MODEL_DTYPE = torch.bfloat16

# "Complex human instruction" prepended to every positive prompt, copied from PixelDiT's stage-3 config.
COMPLEX_HUMAN_INSTRUCTION = "\n".join([
    'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:',
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
    "User Prompt: ",
])


def encode_prompts(prompts, is_positive, tokenizer, text_encoder, instruction_length):
    """
    Encodes a list of prompts the way PixelDiT's inference does.

    Args:
        prompts: list of strings.
        is_positive: True for positive prompts (instruction prefix, first + last PROMPT_LENGTH - 1 tokens kept),
            False for negative prompts (plain, PROMPT_LENGTH tokens).
        tokenizer: Gemma tokenizer (right padding).
        text_encoder: Gemma2Model.
        instruction_length: number of tokens of COMPLEX_HUMAN_INSTRUCTION, including the start token.

    Returns:
        (features, word_masks): [len(prompts), PROMPT_LENGTH, text dim] CPU tensor, and [len(prompts),
        PROMPT_LENGTH] bool CPU tensor marking the prompt's own words (no start token, instruction or padding).
    """
    if is_positive:
        prompts = [COMPLEX_HUMAN_INSTRUCTION + prompt for prompt in prompts]
    max_length = instruction_length + PROMPT_LENGTH - 2 if is_positive else PROMPT_LENGTH
    tokens = tokenizer(prompts, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    tokens = tokens.to(text_encoder.device)
    features = text_encoder(tokens.input_ids, attention_mask=tokens.attention_mask).last_hidden_state
    # The prompt's first word starts after the start token and, for positive prompts, after the instruction. The
    # instruction's trailing space merges into that first word's token, so it is counted without it.
    first_word_index = len(tokenizer.encode(COMPLEX_HUMAN_INSTRUCTION.rstrip())) if is_positive else 1
    word_masks = tokens.attention_mask.bool()
    word_masks[:, :first_word_index] = False
    if is_positive:
        kept_positive_tokens = [0] + list(range(-PROMPT_LENGTH + 1, 0))
        features, word_masks = features[:, kept_positive_tokens], word_masks[:, kept_positive_tokens]
    return features.cpu(), word_masks.cpu()


def prompt_cache_path(prompt, is_positive):
    """
    Args:
        prompt: prompt string.
        is_positive: whether it is encoded as a positive prompt (with the instruction).

    Returns:
        Path of the prompt's cache file, named by a hash of everything its encoding depends on.
    """
    kind = COMPLEX_HUMAN_INSTRUCTION if is_positive else "negative prompt"
    cache_key = "\n".join([TEXT_ENCODER_REPO_ID, str(PROMPT_LENGTH), kind, prompt])
    return PROMPT_CACHE_DIR / f"{hashlib.sha256(cache_key.encode()).hexdigest()}.pt"


@torch.no_grad()
def set_text_embeddings(latents_arr, device):
    """
    Encodes every tile's prompts with Gemma and stores them on the tiles, then frees Gemma.

    Positive prompts get the complex human instruction prefix and keep the first token plus the last
    PROMPT_LENGTH - 1 tokens; negative prompts are encoded plainly to PROMPT_LENGTH tokens. Padding tokens are
    kept (the model was trained without a text mask). This mirrors Mix_n_match's encode_all_prompts.

    Args:
        latents_arr: the LatentClass tiles; each one's `prompt` / `negative_prompt` is encoded and its
            `text_embeddings` is set to a [2, PROMPT_LENGTH, text dim] CPU tensor (index 0 negative, 1 positive).
        device: torch device to run Gemma on.
    """
    # Keys are (prompt, is_positive); values are {"features", "word_mask"} dicts.
    unique_keys = {(latent.prompt, True) for latent in latents_arr}
    unique_keys |= {(latent.negative_prompt, False) for latent in latents_arr}
    encoded = {key: torch.load(prompt_cache_path(*key)) for key in unique_keys if prompt_cache_path(*key).exists()}
    missing_keys = sorted(unique_keys - encoded.keys())
    print(f"Prompt cache: {len(encoded)} of {len(unique_keys)} prompts cached")

    if missing_keys:
        if not TEXT_ENCODER_DIR.exists():
            print(f"Downloading {TEXT_ENCODER_REPO_ID} to {TEXT_ENCODER_DIR} (one time only)...")
            snapshot_download(TEXT_ENCODER_REPO_ID, local_dir=TEXT_ENCODER_DIR,
                              allow_patterns=TEXT_ENCODER_FILE_PATTERNS)
        tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER_DIR)
        tokenizer.padding_side = "right"
        text_encoder = Gemma2Model.from_pretrained(TEXT_ENCODER_DIR, dtype=MODEL_DTYPE).to(device).eval()
        instruction_length = len(tokenizer.encode(COMPLEX_HUMAN_INSTRUCTION))
        PROMPT_CACHE_DIR.mkdir(exist_ok=True)
        for is_positive in (False, True):
            prompts = [prompt for prompt, key_is_positive in missing_keys if key_is_positive == is_positive]
            for start in range(0, len(prompts), ENCODE_BATCH_SIZE):
                batch = prompts[start:start + ENCODE_BATCH_SIZE]
                features, word_masks = encode_prompts(batch, is_positive, tokenizer, text_encoder, instruction_length)
                for prompt, prompt_features, word_mask in zip(batch, features, word_masks, strict=True):
                    entry = {"features": prompt_features.clone(), "word_mask": word_mask.clone()}
                    torch.save(entry, prompt_cache_path(prompt, is_positive))
                    encoded[(prompt, is_positive)] = entry
        # The transformer is loaded next, and the two do not fit on an 8 GB GPU together.
        del text_encoder
        gc.collect()
        torch.cuda.empty_cache()

    for latent in latents_arr:
        latent.text_embeddings = torch.stack([
            encoded[(latent.negative_prompt, False)]["features"],
            encoded[(latent.prompt, True)]["features"],
        ])
