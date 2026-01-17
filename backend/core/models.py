import torch
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionImg2ImgPipeline, UNet2DConditionModel
from .config import device

# --- MODEL LOADING ---

print("⏳ Loading AI Models (This takes ~2 mins)...")

try:
    # A. Load the Shield Engine (Inpainting)
    # We need this specific model for the 'Extreme' protection calculation
    shield_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    shield_pipe.set_progress_bar_config(disable=True)
    
    # B. Load the Verification Engine (Standard UNet)
    # We download ONLY the UNet from standard SD 1.5 to fix the channel error
    print("⏳ Loading Verification Brain...")
    std_unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        subfolder="unet", 
        torch_dtype=torch.float16
    ).to(device)

    # C. Build Verification Pipeline (Hybrid)
    # We reuse the VAE and Text Encoder from shield_pipe to save 3GB VRAM
    verify_pipe = StableDiffusionImg2ImgPipeline(
        vae=shield_pipe.vae,
        text_encoder=shield_pipe.text_encoder,
        tokenizer=shield_pipe.tokenizer,
        unet=std_unet,              # <--- USING STANDARD UNET HERE
        scheduler=shield_pipe.scheduler,
        safety_checker=None,
        feature_extractor=None
    ).to(device)
    
    # D. VAE Reference for Simple Shield
    vae = shield_pipe.vae
    vae.requires_grad_(False)
    
    print("✅ Hybrid Architecture Loaded Successfully")

except Exception as e:
    print(f"❌ Model Error: {e}")
    # In a real app we might raise the error or exit, but preserving original logic implies continuing
    # However, if models fail, the app essentially fails. 
    # Attempting to define them as None to avoid NameErrors if imports fail later
    shield_pipe = None
    verify_pipe = None
    vae = None
