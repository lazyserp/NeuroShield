# ==========================================
#  IMAGE IMMUNIZER: PRODUCTION SERVER
#  (Fixed: Dual-UNet Architecture)
# ==========================================

import subprocess
import sys
import os

# --- 1. ROBUST INSTALLER ---
def install_packages():
    print("⏳ Checking libraries...")
    packages = "fastapi uvicorn python-multipart pyngrok nest_asyncio torch diffusers transformers accelerate scipy ftfy"
    subprocess.check_call([sys.executable, "-m", "pip", "install", *packages.split()])
    print("✅ Libraries Installed.")

try:
    import pyngrok
    import diffusers
    import fastapi
except ImportError:
    install_packages()

# --- 2. IMPORTS ---
import io
import torch
import uvicorn
import nest_asyncio
import gc
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionImg2ImgPipeline, UNet2DConditionModel
from torchvision import transforms

# --- 3. CONFIG & AUTH ---
token = os.environ.get("NGROK_TOKEN")
if not token:
    token = input("Paste Ngrok Token: ")
ngrok.set_auth_token(token)

nest_asyncio.apply()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Engine Starting on {device}...")

# --- 4. MODEL LOADING (The Fix) ---
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

# --- 5. ATTACK ALGORITHMS ---

def attack_simple(X, model, eps=0.05, steps=40):
    delta = torch.zeros_like(X).uniform_(-eps, eps).to(device)
    delta.requires_grad = True
    step_size = 0.01
    
    for _ in range(steps):
        adv_image = X + delta
        latents = model(adv_image).latent_dist.mean
        loss = latents.norm()
        
        grad = torch.autograd.grad(loss, delta)[0]
        delta.data = delta.data - step_size * grad.sign()
        delta.data = torch.clamp(delta.data, -eps, eps)
        delta.data = torch.clamp(X + delta.data, -1, 1) - X
        if delta.grad is not None: delta.grad.zero_()
            
    return X + delta

def attack_extreme(pipe, image_tensor, steps=20, eps=0.04):
    X_adv = image_tensor.clone().detach()
    X_adv.requires_grad = True
    target_image = torch.zeros_like(X_adv).to(device).half()
    step_size = 0.01

    for i in range(steps):
        torch.set_grad_enabled(True)
        X_adv.requires_grad_(True)
        
        # Inpainting Forward Pass
        latents = pipe.vae.encode(X_adv).latent_dist.sample() * 0.18215
        noise = torch.randn_like(latents)
        timesteps = torch.tensor([500], device=device)
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
        
        # Important: pass dummy mask arguments for the Inpainting UNet
        # We simulate a "no-op" to get gradients
        batch_size = latents.shape[0]
        mask = torch.zeros((batch_size, 1, latents.shape[2], latents.shape[3]), device=device, dtype=latents.dtype)
        masked_latents = latents # If mask is 0, masked_latents is just latents
        
        # Concatenate inputs as Inpainting UNet expects 9 channels
        latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)
        
        noise_pred = pipe.unet(latent_model_input, timesteps, encoder_hidden_states=torch.zeros((1, 77, 768), device=device).half())[0]
        
        loss = (X_adv - target_image).norm() 
        grad = torch.autograd.grad(loss, [X_adv])[0]
        
        X_adv = X_adv - step_size * grad.sign()
        X_adv = torch.minimum(torch.maximum(X_adv, image_tensor - eps), image_tensor + eps)
        X_adv.data = torch.clamp(X_adv, min=-1, max=1)
        
        if i % 5 == 0: torch.cuda.empty_cache()
            
    return X_adv

# --- 6. API ENDPOINTS ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

def image_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf

@app.post("/immunize")
async def immunize_endpoint(file: UploadFile = File(...), mode: str = Form("simple")):
    print(f"🛡️ Immunize Request: {file.filename} | Mode: {mode}")
    image_bytes = await file.read()
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    W, H = original_image.size
    
    max_dim = 1024 if mode == "extreme" else 2048
    if max(W, H) > max_dim:
        scale = max_dim / max(W, H)
        original_image = original_image.resize((int(W*scale), int(H*scale)), Image.LANCZOS)
    
    new_W, new_H = (original_image.width // 8) * 8, (original_image.height // 8) * 8
    input_image = original_image.resize((new_W, new_H))
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    X = transform(input_image).unsqueeze(0).to(device).half()

    if mode == "extreme":
        X_adv = attack_extreme(shield_pipe, X, steps=25)
    else:
        X_adv = attack_simple(X, vae.encode, steps=40)

    X_adv = X_adv.float().detach().cpu()
    X_adv = (X_adv / 2 + 0.5).clamp(0, 1)
    img_array = X_adv[0].permute(1, 2, 0).numpy()
    protected_image = Image.fromarray((img_array * 255).astype(np.uint8))
    
    if protected_image.size != (W, H):
        protected_image = protected_image.resize((W, H), Image.LANCZOS)
    
    torch.cuda.empty_cache()
    return StreamingResponse(image_to_bytes(protected_image), media_type="image/png")

@app.post("/verify")
async def verify_endpoint(file: UploadFile = File(...), prompt: str = Form(...)):
    print(f"🤖 Verification Attack: '{prompt}'")
    
    image_bytes = await file.read()
    init_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Resize to standard SD size for verification speed
    init_image = init_image.resize((512, 512)) 
    
    with torch.autocast("cuda"), torch.inference_mode():
        result = verify_pipe(
            prompt=prompt, 
            image=init_image, 
            strength=0.6,          # High strength = AI tries hard to edit
            guidance_scale=7.5, 
            num_inference_steps=20
        ).images[0]
    
    torch.cuda.empty_cache()
    return StreamingResponse(image_to_bytes(result), media_type="image/png")

@app.get("/")
def home(): 
    return {"status": "Online", "engines": "Hybrid (Inpaint+Standard)"}

# --- 7. START ---
ngrok.kill()
try:
    public_url = ngrok.connect(8000).public_url
    print(f"\n🔗 SERVER URL: {public_url}")
    print("👉 Update your Frontend with this new URL")
except Exception as e:
    print(f"Ngrok Error: {e}")

config = uvicorn.Config(app, host="0.0.0.0", port=8000)
server = uvicorn.Server(config)
await server.serve()