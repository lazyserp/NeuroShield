from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import torch
import numpy as np
from torchvision import transforms

from ..core.config import device
from ..core.models import shield_pipe, verify_pipe, vae
from ..core.attacks import attack_simple, attack_extreme

router = APIRouter()

# --- HELPER ---
def image_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf

# --- ENDPOINTS ---

@router.get("/")
def home(): 
    return {"status": "Online", "engines": "Hybrid (Inpaint+Standard)"}

@router.post("/immunize")
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
        # shield_pipe is instantiated in models.py
        if shield_pipe is None: 
            return {"error": "Model not loaded"}
        X_adv = attack_extreme(shield_pipe, X, steps=25)
    else:
        if vae is None:
            return {"error": "Model not loaded"}
        X_adv = attack_simple(X, vae.encode, steps=40)

    X_adv = X_adv.float().detach().cpu()
    X_adv = (X_adv / 2 + 0.5).clamp(0, 1)
    img_array = X_adv[0].permute(1, 2, 0).numpy()
    protected_image = Image.fromarray((img_array * 255).astype(np.uint8))
    
    if protected_image.size != (W, H):
        protected_image = protected_image.resize((W, H), Image.LANCZOS)
    
    torch.cuda.empty_cache()
    return StreamingResponse(image_to_bytes(protected_image), media_type="image/png")

@router.post("/verify")
async def verify_endpoint(file: UploadFile = File(...), prompt: str = Form(...)):
    print(f"🤖 Verification Attack: '{prompt}'")
    
    if verify_pipe is None:
        return {"error": "Model not loaded"}

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
