# ==========================================
#  NEUROSHIELD: COMPLEX DIFFUSION ATTACK
#  (Based on your 'demo_complex_attack_inpainting.ipynb')
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
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok
from diffusers import StableDiffusionInpaintPipeline
from torchvision import transforms
from tqdm import tqdm

# 3. AUTH
print("="*60)
NGROK_TOKEN = input("Paste Ngrok Token: ")
ngrok.set_auth_token(NGROK_TOKEN)
print("✅ Token Set.")

# 4. SETUP
ngrok.kill()
try:
    !fuser -k 8000/tcp > /dev/null 2>&1
except:
    pass

nest_asyncio.apply()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Engine Starting on {device}...")

# 5. LOAD MODEL (Full Inpainting Pipeline)
try:
    # We load the FULL pipeline, not just VAE, because the complex attack targets the UNet
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe = pipe.to(device)
    # Disable progress bar for internal steps
    pipe.set_progress_bar_config(disable=True)
    print("✅ Inpainting Pipeline Loaded")
except Exception as e:
    print(f"❌ Model Error: {e}")

# 6. APP
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- COMPLEX ATTACK LOGIC (EXACT PORT FROM NOTEBOOK) ---

def attack_forward(self, prompt, masked_image, mask, height=512, width=512, num_inference_steps=4, guidance_scale=7.5):
    """
    Differentiable forward pass of the UNet (The 'Complex' part).
    Simulates the diffusion process to find gradients.
    """
    # 1. Encode Text
    text_inputs = self.tokenizer(
        prompt, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt"
    )
    text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]

    uncond_input = self.tokenizer(
        [""], padding="max_length", max_length=text_inputs.input_ids.shape[-1], return_tensors="pt"
    )
    uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # 2. Prepare Latents
    latents = torch.randn((1, 4, height // 8, width // 8), device=self.device, dtype=text_embeddings.dtype)
    latents = latents * self.scheduler.init_noise_sigma
    
    # 3. Prepare Mask & Image
    mask_processed = torch.nn.functional.interpolate(mask, size=(height // 8, width // 8))
    mask_processed = torch.cat([mask_processed] * 2)

    masked_image_latents = self.vae.encode(masked_image).latent_dist.sample()
    masked_image_latents = 0.18215 * masked_image_latents
    masked_image_latents = torch.cat([masked_image_latents] * 2)

    # 4. Differentiable Diffusion Loop (Shortened to 'num_inference_steps')
    self.scheduler.set_timesteps(num_inference_steps)
    
    for t in self.scheduler.timesteps:
        # Expand for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = torch.cat([latent_model_input, mask_processed, masked_image_latents], dim=1)
        
        # Predict Noise
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        
        # Guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Step
        latents = self.scheduler.step(noise_pred, t, latents).prev_sample

    # Decode
    latents = 1 / 0.18215 * latents
    image = self.vae.decode(latents).sample
    return image

def compute_grad(cur_mask, cur_masked_image, prompt, target_image, height, width):
    torch.set_grad_enabled(True)
    cur_masked_image.requires_grad_(True)
    
    # Run the differentiable forward pass
    image_nat = attack_forward(
        pipe,
        prompt=prompt,
        masked_image=cur_masked_image,
        mask=cur_mask,
        height=height,
        width=width,
        num_inference_steps=4 # Fast approximation (from your notebook)
    )
    
    # Loss: Make the result look like the 'target' (Zero/Gray)
    loss = (image_nat - target_image).norm(p=2)
    
    # Grad: Only update the UNMASKED context (which is everything in our case)
    grad = torch.autograd.grad(loss, [cur_masked_image])[0] * (1 - cur_mask)
    
    return grad, loss.item()

def super_linf(image_bytes, steps=50, eps=0.05): # Reduced eps from 0.1 to 0.05 to reduce patterns
    # A. Load & Resize (High-Res)
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    W, H = original_image.size
    print(f"📸 Input: {W}x{H}")

    # Resize to multiple of 8
    new_W = (W // 8) * 8
    new_H = (H // 8) * 8
    if new_W != W or new_H != H:
        input_image = original_image.resize((new_W, new_H), Image.LANCZOS)
    else:
        input_image = original_image

    # B. Prep Tensors
    # Normalize to [-1, 1]
    X = transforms.ToTensor()(input_image).unsqueeze(0).to(device)
    X = 2.0 * X - 1.0
    X = X.half() # Use FP16 for speed/memory
    
    # Mask: All Zeros (Treat whole image as context to protect)
    cur_mask = torch.zeros((1, 1, new_H, new_W)).to(device).half()
    
    # Target: All Zeros (The Void)
    target_image = torch.zeros_like(X).to(device).half()
    
    # C. PGD Loop (L-Infinity)
    X_adv = X.clone().detach()
    step_size = 0.01 # Standard step size
    
    print("⚔️ Running Complex Diffusion Attack...")
    for i in range(steps):
        # 1. Compute Gradient (Heavy Step)
        grad, loss_val = compute_grad(cur_mask, X_adv, prompt="", target_image=target_image, height=new_H, width=new_W)
        
        # 2. Update
        X_adv = X_adv - grad.sign() * step_size
        
        # 3. Project
        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=-1, max=1)
        
        # Cleanup
        if i % 10 == 0: print(f"Step {i}/{steps} | Loss: {loss_val:.2f}")
        torch.cuda.empty_cache()

    # D. Save
    X_adv = X_adv.float().detach().cpu()
    X_adv = (X_adv / 2 + 0.5).clamp(0, 1)
    
    img_array = X_adv[0].permute(1, 2, 0).numpy()
    protected_image = Image.fromarray((img_array * 255).astype(np.uint8))
    
    if protected_image.size != (W, H):
        protected_image = protected_image.resize((W, H), Image.LANCZOS)

    img_byte_arr = io.BytesIO()
    protected_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

@app.post("/immunize")
async def immunize_endpoint(file: UploadFile = File(...)):
    print(f"⚡ Processing: {file.filename}")
    image_bytes = await file.read()
    return StreamingResponse(super_linf(image_bytes), media_type="image/png")

@app.get("/")
def home():
    return {"status": "Complex Engine Online"}

# 7. START
public_url = ngrok.connect(8000).public_url
print(f"\n🔗 PUBLIC URL: {public_url}")
print("👉 Copy this into your Frontend")
print("="*60 + "\n")

config = uvicorn.Config(app, host="0.0.0.0", port=8000)
server = uvicorn.Server(config)
await server.serve()