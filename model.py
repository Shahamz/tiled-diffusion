import gc
import logging

import PIL.Image
import numpy as np
import torch
from diffusers import (AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler, DDPMScheduler, DDIMScheduler,
                       DPMSolverMultistepScheduler)
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from config import MODEL_ID_SD_1_5
from latent_handler import LatentHandler
from pixeldit.macros import PATCH_SIZE_PIXELS
from pixeldit.pixeldit_model import load_pixeldit
from pixeldit.text_encoder import MODEL_DTYPE, set_text_embeddings
from utils import retrieve_latents, retrieve_timesteps, get_timesteps, randn_tensor, preprocess, generate_graph_groups

# PixelDiT sampling constants, matching Mix_n_match/pipeline_mix_n_match.py.
PIXEL_CHANNELS = 3          # PixelDiT denoises the image itself, not a 4 channel latent
NUM_TRAIN_TIMESTEPS = 1000  # the model's time input is sigma * NUM_TRAIN_TIMESTEPS
SOLVER_ORDER = 2            # order of the DPM-Solver++ multistep sampler PixelDiT's inference uses


class SDLatentTiling:
    def __init__(self, model_id=MODEL_ID_SD_1_5, scheduler="ddpm"):
        self.model_id = model_id
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_id,
            subfolder='tokenizer',
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id,
            subfolder='text_encoder',
            # use_safetensors=True,
        ).to('cuda')

        self.unet = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder='unet',
            # use_safetensors=True,
        ).to('cuda')

        self.vae = AutoencoderKL.from_pretrained(
            model_id,
            subfolder='vae',
            # use_safetensors=True,
        ).to('cuda')

        if scheduler == "ddpm":
            self.scheduler = DDPMScheduler.from_pretrained(
                model_id,
                subfolder='scheduler',
            )
        elif scheduler == "ddim":
            self.scheduler = DDIMScheduler.from_pretrained(
                model_id,
                subfolder='scheduler',
            )
        else:  # Euler
            self.scheduler = EulerDiscreteScheduler.from_pretrained(
                model_id,
                subfolder='scheduler',
            )
        logging.warning("Finished loading models..")

    def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            init_latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                init_latents = [
                    retrieve_latents(self.vae.encode(image[i: i + 1]), generator=generator[i])
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = retrieve_latents(self.vae.encode(image), generator=generator)

            init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            logging.warning(f"len(prompt) != len(image)\n{deprecation_message}")
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # get latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents

        return latents

    def get_text_embeddings(self, prompts):
        tokenized_prompts = self.tokenizer(
            prompts,
            max_length=self.tokenizer.model_max_length,  # Size we need (77)
            padding='max_length',  # Apply padding if necessary
            truncation=True,  # Apply truncation if necessary
            return_tensors='pt',
        )
        with torch.no_grad():
            embeddings = self.text_encoder(tokenized_prompts.input_ids.to('cuda'))[0]
        return embeddings

    def __call__(self, latents_arr, negative_prompt="", inference_steps=30, seed=42,
                 cfg_scale=12.5, height=512, width=512, max_width=10, max_replica_width=5, strength=0.8,
                 device='cpu'):
        for latent in latents_arr:
            latent.set_text_embs(tokenizer=self.tokenizer, text_encoder=self.text_encoder)
        generator = torch.Generator(device='cuda')
        generator.manual_seed(seed)
        for latent in latents_arr:
            if latent.source_image is not None:
                image = preprocess(latent.source_image)
                timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, inference_steps, device, None)
                timesteps, num_inference_steps = get_timesteps(self.scheduler, num_inference_steps, strength, device)
                latent_timestep = timesteps[:1].repeat(1)
                latents = self.prepare_latents(
                    image,
                    latent_timestep,
                    1,
                    1,
                    latent.text_embeddings.dtype,
                    device,
                    generator,
                )
                B, F, H, W = latents.size()
                if latent.is_xy():
                    random_x_tensor = torch.randn(B, F, H, max_width, generator=generator, device='cuda')
                    latents_new = torch.cat([random_x_tensor, latents, random_x_tensor], dim=3)
                    B, F, H, W = latents_new.size()
                    random_y_tensor = torch.randn(B, F, max_width, W, generator=generator, device='cuda')
                    final_latents = torch.cat([random_y_tensor, latents_new, random_y_tensor], dim=2)
                elif latent.is_x():
                    random_x_tensor = torch.randn(B, F, H, max_width, generator=generator, device='cuda')
                    final_latents = torch.cat([random_x_tensor, latents, random_x_tensor], dim=3)
                elif latent.is_y():
                    random_y_tensor = torch.randn(B, F, max_width, W, generator=generator, device='cuda')
                    final_latents = torch.cat([random_y_tensor, latents, random_y_tensor], dim=2)
                else:
                    final_latents = latents

                latent.pre_latent = final_latents
                latent.clone_post_latents()
            else:
                latent.set_latents(generator=generator, max_width=max_width)

        self.scheduler.set_timesteps(inference_steps)
        graph_groups = generate_graph_groups(latents_arr)
        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            logging.warning(f"Running step {i} of {inference_steps}")
            logging.warning(f"Applying Similarity Constraint")
            latents_arr = LatentHandler.apply_similarity_constraint(latents_arr, i,
                                                                    groups=graph_groups,
                                                                    max_width=max_width,
                                                                    max_replica_width=max_replica_width)
            for latent in latents_arr:
                latent.clone_post_latents()
            logging.warning(f"Tiling latents")
            latents_arr = LatentHandler.tile(latents_arr, i, groups=graph_groups, max_width=max_width)
            for latent in latents_arr:
                torch.cuda.empty_cache()
                gc.collect()
                latent_model_input = torch.cat([latent.post_latent] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
                with torch.no_grad():
                    # The U-Net is asked to make a prediction of the amount of noise in the tensor
                    noise_pred = self.unet(latent_model_input, t,
                                           encoder_hidden_states=latent.text_embeddings).sample

                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

                # A new tensor is generated by subtracting the amount of noise we have previously calculated
                # This is the process that cleans the noise until all the scheduler steps have finished
                latent_new = self.scheduler.step(noise_pred, t, latent.post_latent).prev_sample
                latent.pre_latent = latent_new
                latent.clone_post_latents()

        logging.warning("Decoding latents and getting images")
        logging.warning(f"Applying Similarity Constraint")
        latents_arr = LatentHandler.apply_similarity_constraint(latents_arr, inference_steps,
                                                                groups=graph_groups,
                                                                max_width=max_width,
                                                                max_replica_width=max_replica_width)
        for latent in latents_arr:
            latent.clone_post_latents()
        logging.warning(f"Tiling latents")
        latents_arr = LatentHandler.tile(latents_arr, inference_steps, groups=graph_groups, max_width=max_width)

        # Add the new function call here
        logging.warning(f"Applying Random Padding Constraint")
        latents_arr = LatentHandler.apply_random_padding_constraint(latents_arr, groups=graph_groups,
                                                                    max_width=max_width)

        for latent in latents_arr:
            torch.cuda.empty_cache()
            gc.collect()
            latents = latent.post_latent / self.vae.config.scaling_factor
            with torch.no_grad():
                images = self.vae.decode(latents).sample
            images = (images / 2 + 0.5).clamp(0, 1)
            image_t = images[0].cpu().detach().numpy()
            img_t_rgb = np.transpose(image_t, (1, 2, 0))
            if latent.is_xy():
                img_t_rgb = img_t_rgb[max_width * 8:-max_width * 8, max_width * 8:-max_width * 8, :]
            elif latent.is_x():
                img_t_rgb = img_t_rgb[:, max_width * 8:-max_width * 8, :]
            elif latent.is_y():
                img_t_rgb = img_t_rgb[max_width * 8:-max_width * 8, :, :]

            latent.image = img_t_rgb

        return latents_arr


class PixelDiTLatentTiling:
    """The same tiling run as SDLatentTiling, denoising with NVIDIA's PixelDiT instead of Stable Diffusion.

    The differences that matter here are all consequences of PixelDiT being a *pixel space* model:

      * a tile's tensor is the 3 channel image itself, at full resolution, so there is no VAE to encode or
        decode through and `set_latents` is called with `downsample=1`;
      * conditioning comes from Gemma-2-2b-it, not CLIP. Gemma (about 5 GB in bfloat16) and the transformer
        (about 2.6 GB) do not fit on an 8 GB card together, so every prompt is encoded and cached first and
        Gemma is freed before the checkpoint is loaded (pixeldit/text_encoder.py);
      * sampling is DPM-Solver++ (order 2) on flow sigmas. That solver is *multistep*: it carries a history and
        a step index, so unlike DDPM it cannot be shared between tiles and every tile gets its own;
      * `max_width` / `max_replica_width` arrive counted in 16 pixel patches and are turned into pixels here,
        so the LatentHandler band copies keep working unchanged.

    The constraint schedule itself -- similarity, tiling, and the closing random padding pass -- is exactly the
    one SDLatentTiling runs.
    """

    def __init__(self, scheduler="ddpm"):
        # PixelDiT samples with flow DPM-Solver++ only, so the scheduler name run.py passes for the Stable
        # Diffusion path has nothing to select here.
        if scheduler is not None:
            logging.warning(f"PixelDiT samples with flow DPM-Solver++ only; --scheduler {scheduler} is ignored")
        # Both models are loaded in __call__, in the order their memory allows.
        self.transformer = None
        self.model_config = None

    def build_scheduler(self, inference_steps, device):
        """Build one tile's sampler: PixelDiT's inference settings, with the checkpoint's flow shift."""
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=NUM_TRAIN_TIMESTEPS,
            solver_order=SOLVER_ORDER,
            algorithm_type="dpmsolver++",
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            flow_shift=self.model_config["scheduler"]["flow_shift"],
        )
        scheduler.set_timesteps(inference_steps, device=device)
        return scheduler

    def predict_velocity(self, latent, sigma, cfg_scale, device):
        """Run the transformer for both guidance passes on one tile and combine them.

        Returns a [1, 3, H, W] float32 flow velocity, the shape of the tile's own tensor.
        """
        # The model takes a stack of images, [batch, S, 3, H, W]; one tile at a time means batch = S = 1.
        images = latent.post_latent.unsqueeze(0).to(MODEL_DTYPE)
        # The exact time sigma * 1000 (scheduler.timesteps are rounded), in the model dtype.
        model_time = torch.full((1,), sigma * NUM_TRAIN_TIMESTEPS, device=device, dtype=MODEL_DTYPE)
        # [2, prompt length, text dim] -> [2, 1, prompt length, text dim]: one prompt per guidance pass.
        text_embeds = latent.text_embeddings.unsqueeze(1).to(device=device, dtype=MODEL_DTYPE)
        # Sequential classifier free guidance: the two passes one after the other give the same result as one
        # batch of two for about half the memory, which is what fits next to the 1.3B checkpoint on 8 GB.
        negative = self.transformer(images, model_time, text_embeds[:1], None)[0].float()
        positive = self.transformer(images, model_time, text_embeds[1:], None)[0].float()
        return negative + cfg_scale * (positive - negative)

    @torch.no_grad()
    def __call__(self, latents_arr, negative_prompt="", inference_steps=30, seed=42,
                 cfg_scale=4.5, height=512, width=512, max_width=16, max_replica_width=5, strength=0.8,
                 device='cpu'):
        for latent in latents_arr:
            if latent.source_image is not None:
                raise NotImplementedError(
                    "The PixelDiT path generates from noise only, so LatentClass.source_image is not supported. "
                    "Use SDLatentTiling for image to image runs."
                )

        # The widths arrive in 16 pixel patches, but the LatentHandler methods slice the tensor itself, which
        # here is the image, so from this point on they are pixels.
        pad = max_width * PATCH_SIZE_PIXELS
        replica = max_replica_width * PATCH_SIZE_PIXELS

        set_text_embeddings(latents_arr, device)
        self.transformer, self.model_config = load_pixeldit(device, MODEL_DTYPE)
        logging.warning("Finished loading models..")

        generator = torch.Generator(device='cuda')
        generator.manual_seed(seed)
        schedulers = []
        for latent in latents_arr:
            latent.set_latents(generator=generator, in_channels=PIXEL_CHANNELS, max_width=pad, downsample=1)
            schedulers.append(self.build_scheduler(inference_steps, device))
        # Every tile's sampler was built the same way, so any of them can drive the loop.
        timesteps, sigmas = schedulers[0].timesteps, schedulers[0].sigmas

        graph_groups = generate_graph_groups(latents_arr)
        for i, t in tqdm(enumerate(timesteps)):
            logging.warning(f"Running step {i} of {inference_steps}")
            logging.warning(f"Applying Similarity Constraint")
            latents_arr = LatentHandler.apply_similarity_constraint(latents_arr, i,
                                                                    groups=graph_groups,
                                                                    max_width=pad,
                                                                    max_replica_width=replica)
            for latent in latents_arr:
                latent.clone_post_latents()
            logging.warning(f"Tiling latents")
            latents_arr = LatentHandler.tile(latents_arr, i, groups=graph_groups, max_width=pad)
            sigma = sigmas[i].item()
            for latent, scheduler in zip(latents_arr, schedulers):
                torch.cuda.empty_cache()
                gc.collect()
                # Flow matching: the model predicts the velocity, noise - image, rather than the noise.
                velocity = self.predict_velocity(latent, sigma, cfg_scale, device)
                latent.pre_latent = scheduler.step(velocity, t, latent.post_latent).prev_sample
                latent.clone_post_latents()

        logging.warning("Getting images")
        logging.warning(f"Applying Similarity Constraint")
        latents_arr = LatentHandler.apply_similarity_constraint(latents_arr, inference_steps,
                                                                groups=graph_groups,
                                                                max_width=pad,
                                                                max_replica_width=replica)
        for latent in latents_arr:
            latent.clone_post_latents()
        logging.warning(f"Tiling latents")
        latents_arr = LatentHandler.tile(latents_arr, inference_steps, groups=graph_groups, max_width=pad)

        logging.warning(f"Applying Random Padding Constraint")
        latents_arr = LatentHandler.apply_random_padding_constraint(latents_arr, groups=graph_groups,
                                                                    max_width=pad)

        for latent in latents_arr:
            torch.cuda.empty_cache()
            gc.collect()
            # Pixel space: the tensor is the image already, so there is no VAE decode, only the [-1, 1] to
            # [0, 1] rescale and the same padding crop the Stable Diffusion path does.
            image_t = ((latent.post_latent[0].float().clamp(-1, 1) + 1) / 2).cpu().numpy()
            img_t_rgb = np.transpose(image_t, (1, 2, 0))
            if latent.is_xy():
                img_t_rgb = img_t_rgb[pad:-pad, pad:-pad, :]
            elif latent.is_x():
                img_t_rgb = img_t_rgb[:, pad:-pad, :]
            elif latent.is_y():
                img_t_rgb = img_t_rgb[pad:-pad, :, :]

            latent.image = img_t_rgb

        return latents_arr
