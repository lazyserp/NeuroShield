# ==========================================
#  NEUROSHIELD: LOW-MEMORY PRODUCTION ENGINE
#  (FP16 Optimized + Encoder Attack)
# ==========================================

# 1. INSTALL
print("⏳ Installing libraries... (Wait ~60s)")
!pip install fastapi uvicorn python-multipart pyngrok nest_asyncio torch diffusers transformers accelerate scipy ftfy > /dev/null 2>&1
print("✅ Installed.\n")

# 2. IMPORTS
import os
import io
import torch
import uvicorn
import nest_asyncio
import gc
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok
from diffusers import AutoencoderKL
from torchvision import transforms

# 3. AUTH & CLEANUP
# Force garbage collection immediately
gc.collect()
torch.cuda.empty_cache()

print("="*60)
NGROK_TOKEN = input("Paste Ngrok Token: ")
ngrok.set_auth_token(NGROK_TOKEN)
print("✅ Token Set.")

# Kill old tunnels
ngrok.kill()
try:
    !fuser -k 8000/tcp > /dev/null 2>&1
except:
    pass

nest_asyncio.apply()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Engine Starting on {device}...")

# 4. LOAD MODEL (FP16 MODE - SAVES 50% RAM)
try:
    # We load in float16 to save memory
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-inpainting", 
        subfolder="vae",
        torch_dtype=torch.float16 
    ).to(device)
    vae.requires_grad_(False)
    vae.enable_slicing() # Slicing handles large images in chunks
    print("✅ Model Loaded (FP16 Low-VRAM Mode)")
except Exception as e:
    print(f"❌ Model Error: {e}")

# 5. APP
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- THE LIGHTWEIGHT ATTACK LOGIC ---

def pgd_attack(X, model, eps=0.06, steps=100):
    # Random Init
    delta = torch.zeros_like(X).uniform_(-eps, eps).to(device)
    delta.requires_grad = True
    
    # We use a simple loop without accumulating history to save RAM
    step_size = 0.02
    
    for i in range(steps):
        # 1. Forward (FP16)
        # We add delta to X. Both are FP16.
        adv_image = X + delta
        latents = model(adv_image).latent_dist.mean
        
        # 2. Loss: Latent Norm (No artifacts)
        loss = latents.norm()
        
        # 3. Backward
        grad = torch.autograd.grad(loss, delta)[0]
        
        # 4. Update
        # Decaying step size
        actual_step = step_size * (1 - i/steps)
        if actual_step < 0.005: actual_step = 0.005
        
        delta.data = delta.data - actual_step * grad.sign()
        
        # 5. Project
        delta.data = torch.clamp(delta.data, -eps, eps)
        delta.data = torch.clamp(X + delta.data, -1, 1) - X
        
        # Zero grad to free graph memory
        if delta.grad is not None:
            delta.grad.zero_()
            
    return X + delta

def process_image(image_bytes):
    # A. Load
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    W, H = original_image.size
    print(f"📸 Input: {W}x{H}")

    # Safety Cap: If image is massive (>2500px), resize to prevent crash
    # 2500px is usually enough for any web use.
    if max(W, H) > 2500:
        scale_factor = 2500 / max(W, H)
        new_W_raw = int(W * scale_factor)
        new_H_raw = int(H * scale_factor)
        print(f"⚠️ Resizing large image to {new_W_raw}x{new_H_raw} for safety")
        original_image = original_image.resize((new_W_raw, new_H_raw), Image.LANCZOS)
        W, H = original_image.size

    # Resize to multiple of 8 (VAE requirement)
    new_W = (W // 8) * 8
    new_H = (H // 8) * 8
    if new_W != W or new_H != H:
        input_image = original_image.resize((new_W, new_H), Image.LANCZOS)
    else:
        input_image = original_image

    # B. Prep Tensors (FP16)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    X = transform(input_image).unsqueeze(0).to(device).half() # <--- Convert to FP16
    
    # C. Run Attack
    X_adv = pgd_attack(X, vae.encode, eps=0.06, steps=80)
    
    # D. Recover
    X_adv = X_adv.float().detach().cpu() # Convert back to Float32 for saving
    X_adv = (X_adv / 2 + 0.5).clamp(0, 1)
    
    img_array = X_adv[0].permute(1, 2, 0).numpy()
    protected_image = Image.fromarray((img_array * 255).astype(np.uint8))

    # Strict Resize Back
    if protected_image.size != (W, H):
        protected_image = protected_image.resize((W, H), Image.LANCZOS)
    
    img_byte_arr = io.BytesIO()
    protected_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    # Cleanup Memory
    del X, X_adv, input_image
    torch.cuda.empty_cache()
    
    return img_byte_arr

@app.post("/immunize")
async def immunize_endpoint(file: UploadFile = File(...)):
    print(f"⚡ Processing: {file.filename}")
    image_bytes = await file.read()
    return StreamingResponse(process_image(image_bytes), media_type="image/png")

@app.get("/")
def home():
    return {"status": "Low-Mem Engine Online"}

# 6. START
public_url = ngrok.connect(8000).public_url
print(f"\n🔗 PUBLIC URL: {public_url}")
print("👉 Copy this into your Frontend")
print("="*60 + "\n")

config = uvicorn.Config(app, host="0.0.0.0", port=8000)
server = uvicorn.Server(config)
await server.serve()