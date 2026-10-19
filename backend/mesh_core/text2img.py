"""Text -> image with Stable Diffusion 1.5 (fits in 4 GB VRAM with fp16)."""
from __future__ import annotations

from typing import List, Optional

import torch
from PIL import Image

from .config import DEVICE, TEXT2IMG_MODEL_ID
from .model_manager import manager
from .presets import NEGATIVE_PROMPT


def _load_sd():
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        TEXT2IMG_MODEL_ID,
        torch_dtype=dtype,
        variant="fp16" if DEVICE == "cuda" else None,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )
    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


manager.register("sd15", _load_sd)


def generate_images(
    prompt: str,
    num_images: int = 1,
    steps: int = 30,
    guidance: float = 7.5,
    seed: Optional[int] = None,
    size: int = 512,
) -> List[Image.Image]:
    pipe = manager.get("sd15")
    images: List[Image.Image] = []
    # one image at a time keeps peak VRAM low
    for i in range(num_images):
        generator = None
        if seed is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(int(seed) + i)
        with torch.inference_mode():
            out = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=size,
                height=size,
                generator=generator,
            )
        images.append(out.images[0])
    return images

