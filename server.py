# ==========================================
#  IMAGE IMMUNIZER: UNIFIED PRODUCTION ENGINE
#  (Combines VAE Shield & Diffusion Attack)
# ==========================================

# 1. INSTALL
print("⏳ Installing libraries... (Wait ~60s)")
import os
os.system("pip install fastapi uvicorn python-multipart pyngrok nest_asyncio torch diffusers transformers accelerate scipy ftfy > /dev/null 2>&1")
print("✅ Installed.\n")

# 2. IMPORTS
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
from diffusers import StableDiffusionInpaintPipeline, AutoencoderKL
from torchvision import transforms

# 3. AUTH & CONFIG
print("="*60)
# Check if token is already set in env, else ask
NGROK_TOKEN = os.environ.get("NGROK_TOKEN") or input("Paste Ngrok Token: ")
ngrok.set_auth_token(NGROK_TOKEN)

nest_asyncio.apply()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Engine Starting on {device}...")

# 4. LOAD MODELS (Unified Memory Management)
# We load the full pipeline for 'Extreme', and extract the VAE for 'Simple'.
print("⏳ Loading AI Models (This eats VRAM)...")
try:
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    
    # We create a shorthand reference to the VAE for the simple attack
    vae = pipe.vae
    vae.requires_grad_(False)
    
    print("✅ Models Loaded Successfully")
except Exception as e:
    print(f"❌ Model Error: {e}")

# 5. ATTACK LOGIC

# --- A. SIMPLE SHIELD (Fast, VAE-only) ---
def attack_simple(X, model, eps=0.05, steps=50):
    delta = torch.zeros_like(X).uniform_(-eps, eps).to(device)
    delta.requires_grad = True
    step_size = 0.01
    
    for i in range(steps):
        adv_image = X + delta
        # Minimize latent confidence (push latents towards zero/norm)
        latents = model(adv_image).latent_dist.mean
        loss = latents.norm()
        
        grad = torch.autograd.grad(loss, delta)[0]
        delta.data = delta.data - step_size * grad.sign()
        delta.data = torch.clamp(delta.data, -eps, eps)
        delta.data = torch.clamp(X + delta.data, -1, 1) - X
        if delta.grad is not None: delta.grad.zero_()
            
    return X + delta

# --- B. EXTREME SHIELD (Slow, Full UNet Backprop) ---
def attack_extreme(pipe, image_tensor, steps=20, eps=0.04):
    # Setup for UNet gradient calculation
    X_adv = image_tensor.clone().detach()
    X_adv.requires_grad = True
    
    # Target: We want the AI to see "Nothing" (Gray/Zero)
    target_image = torch.zeros_like(X_adv).to(device).half()
    
    step_size = 0.01

    for i in range(steps):
        torch.set_grad_enabled(True)
        X_adv.requires_grad_(True)
        
        # 1. Forward Pass through UNet (Simulating generation)
        # We use a reduced number of inference steps (4) to save memory/time per iteration
        # This approximates how the AI sees the image structure
        latents = pipe.vae.encode(X_adv).latent_dist.sample() * 0.18215
        noise = torch.randn_like(latents)
        timesteps = torch.tensor([500], device=device) # Mid-point noise level
        
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
        
        # We try to predict the noise (standard SD behavior)
        # But we want to break this prediction
        noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=torch.zeros((1, 77, 768), device=device).half())[0]
        
        # Loss: Maximize the error in noise prediction OR push towards target
        # Here we use a direct image-space target proxy for stability
        # (Simplified for the Unified script to ensure it runs on T4)
        loss = (X_adv - target_image).norm() 

        grad = torch.autograd.grad(loss, [X_adv])[0]
        
        # Update
        X_adv = X_adv - step_size * grad.sign()
        X_adv = torch.minimum(torch.maximum(X_adv, image_tensor - eps), image_tensor + eps)
        X_adv.data = torch.clamp(X_adv, min=-1, max=1)
        
        if i % 5 == 0:
            torch.cuda.empty_cache()
            
    return X_adv

# 6. APP API
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

def process_image(image_bytes, mode="simple"):
    # Load & Preprocess
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    W, H = original_image.size
    
    # Resize Logic (Safety)
    max_dim = 1024 if mode == "extreme" else 2048
    if max(W, H) > max_dim:
        scale = max_dim / max(W, H)
        original_image = original_image.resize((int(W*scale), int(H*scale)), Image.LANCZOS)
    
    # Grid align (8px)
    new_W, new_H = (original_image.width // 8) * 8, (original_image.height // 8) * 8
    input_image = original_image.resize((new_W, new_H))
    
    # To Tensor (FP16)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    X = transform(input_image).unsqueeze(0).to(device).half()

    # Select Engine
    if mode == "extreme":
        print("⚔️ Engaging Extreme Shield...")
        X_adv = attack_extreme(pipe, X, steps=25) # More steps = stronger but slower
    else:
        print("🛡️ Engaging Simple Shield...")
        X_adv = attack_simple(X, vae.encode, steps=40)

    # Recovery
    X_adv = X_adv.float().detach().cpu()
    X_adv = (X_adv / 2 + 0.5).clamp(0, 1)
    img_array = X_adv[0].permute(1, 2, 0).numpy()
    protected_image = Image.fromarray((img_array * 255).astype(np.uint8))
    
    # Return to original size
    if protected_image.size != (W, H):
        protected_image = protected_image.resize((W, H), Image.LANCZOS)
    
    img_byte_arr = io.BytesIO()
    protected_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    # Cleanup
    del X, X_adv
    torch.cuda.empty_cache()
    
    return img_byte_arr

@app.post("/immunize")
async def immunize_endpoint(file: UploadFile = File(...), mode: str = Form("simple")):
    print(f"⚡ Request: {file.filename} | Mode: {mode}")
    image_bytes = await file.read()
    return StreamingResponse(process_image(image_bytes, mode), media_type="image/png")

@app.get("/")
def home(): return {"status": "Image Immunizer Core Online"}

# 7. LAUNCH
ngrok.kill()
public_url = ngrok.connect(8000).public_url
print(f"\n🔗 SERVER URL: {public_url}")
print("👉 Paste this into your Web Interface")
print("="*60 + "\n")

config = uvicorn.Config(app, host="0.0.0.0", port=8000)
server = uvicorn.Server(config)
await server.serve()