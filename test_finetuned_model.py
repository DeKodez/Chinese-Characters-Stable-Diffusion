"""
Test the fine-tuned LoRA model by generating Chinese characters.
"""

import argparse
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from PIL import Image


def load_finetuned_pipeline(base_model: str, lora_path: str, device: str = "cuda"):
    """Load base model and apply LoRA weights."""
    print(f"Loading base model: {base_model}")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    
    print(f"Loading LoRA weights from: {lora_path}")
    pipe.unet.load_attn_procs(lora_path)
    
    pipe = pipe.to(device)
    return pipe


def generate_character(pipe, prompt: str, negative_prompt: str = "", num_inference_steps: int = 50, guidance_scale: float = 7.5, seed: int = None):
    """Generate a Chinese character image."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    
    return image


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned Chinese character model")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5", help="Base Stable Diffusion model")
    parser.add_argument("--prompt", type=str, required=True, help="Generation prompt")
    parser.add_argument("--output", type=str, default="generated_char.png", help="Output image path")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load pipeline
    pipe = load_finetuned_pipeline(args.base_model, args.lora_path, device)
    
    # Generate
    print(f"Generating with prompt: '{args.prompt}'")
    image = generate_character(
        pipe,
        args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed
    )
    
    # Save
    image.save(args.output)
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
