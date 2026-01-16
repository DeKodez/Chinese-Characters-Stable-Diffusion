"""
Fine-tune Stable Diffusion using LoRA for Chinese character generation.
Uses the diffusers library with PEFT (Parameter-Efficient Fine-Tuning).
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import accelerate
from accelerate import Accelerator
from tqdm.auto import tqdm


class ChineseCharDataset(Dataset):
    """Dataset for Chinese character images with captions."""
    
    def __init__(self, jsonl_path: str, images_dir: str, tokenizer, size: int = 512):
        self.images_dir = Path(images_dir)
        self.size = size
        self.tokenizer = tokenizer
        
        # Load JSONL file
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        print(f"Loaded {len(self.data)} training examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image_path = self.images_dir / item['image_path'] if not Path(item['image_path']).is_absolute() else Path(item['image_path'])
        if not image_path.exists():
            # Try relative to images_dir
            image_path = self.images_dir / Path(item['image_path']).name
        
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.size, self.size), Image.LANCZOS)
        
        # Get caption
        caption = item.get('caption', item.get('character', ''))
        
        # Tokenize caption
        inputs = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "pixel_values": self._transform_image(image),
            "input_ids": inputs.input_ids.flatten(),
        }
    
    def _transform_image(self, image):
        """Transform PIL image to tensor."""
        # Simple normalization: convert to tensor and normalize to [-1, 1]
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
        ])
        return transform(image)


def collate_fn(examples):
    """Collate function for DataLoader."""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion with LoRA")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to JSONL dataset file")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing character images")
    parser.add_argument("--output_dir", type=str, default="./lora_checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--pretrained_model", type=str, default="runwayml/stable-diffusion-v1-5", help="Base model")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help="LR scheduler type")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="LR warmup steps")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Mixed precision")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer and text encoder
    print("Loading tokenizer and text encoder...")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder")
    
    # Load UNet
    print("Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet")
    
    # Setup LoRA
    print("Setting up LoRA...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    
    # Load scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")
    
    # Freeze text encoder
    text_encoder.requires_grad_(False)
    
    # Create dataset and dataloader
    print("Creating dataset...")
    train_dataset = ChineseCharDataset(args.dataset_path, args.images_dir, tokenizer, size=args.resolution)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )
    
    # Setup learning rate scheduler
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps * args.gradient_accumulation_steps,
    )
    
    # Prepare with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    text_encoder = accelerator.prepare(text_encoder)
    
    # Training loop
    print(f"\nStarting training for {args.num_train_epochs} epochs...")
    print(f"Total steps: {max_train_steps}")
    print(f"Batch size: {args.train_batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.train_batch_size * args.gradient_accumulation_steps}")
    
    global_step = 0
    
    for epoch in range(args.num_train_epochs):
        unet.train()
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.num_train_epochs}")
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = batch["pixel_values"] * 2.0 - 1.0  # Normalize to [-1, 1]
                
                # Sample noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
                timesteps = timesteps.long()
                
                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                
                # Predict noise
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Compute loss
                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # Backward pass
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if global_step % args.save_steps == 0:
                    if accelerator.is_main_process:
                        checkpoint_path = output_dir / f"checkpoint-{global_step}"
                        checkpoint_path.mkdir(exist_ok=True)
                        unet.save_pretrained(checkpoint_path)
                        print(f"\nSaved checkpoint to {checkpoint_path}")
                
                progress_bar.set_postfix({"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]})
        
        progress_bar.close()
    
    # Save final checkpoint
    if accelerator.is_main_process:
        final_path = output_dir / "final"
        final_path.mkdir(exist_ok=True)
        unet.save_pretrained(final_path)
        print(f"\nTraining complete! Final checkpoint saved to {final_path}")


if __name__ == "__main__":
    main()
