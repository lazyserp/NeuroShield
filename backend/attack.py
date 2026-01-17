import torch
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms
import numpy as np
import io

# 1. SETUP ENGINE
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"NeuroShield High-Res Core Online. Engine: {device}")

# Load VAE (The Encoder)
try:
    # We use the standard SD v1.5 VAE
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device)
    vae.requires_grad_(False)
    vae.enable_slicing() # optimization to save memory on large images
except Exception as e:
    print(f"Error loading model: {e}")

def immunize_image(image_bytes, epsilon=0.03, steps=50):
    # 2. LOAD IMAGE PRESERVING SIZE
    # Don't force resize to 512x512!
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    W, H = original_image.size
    
    # 3. HANDLE VAE CONSTRAINTS (Multiples of 8)
    # The VAE compresses the image by 8x, so dimensions must be divisible by 8.
    # We resize to the closest multiple of 8 (minimal change, invisible to eye).
    new_W = (W // 8) * 8
    new_H = (H // 8) * 8
    
    # Only resize if necessary (to fit VAE requirements)
    if new_W != W or new_H != H:
        input_image = original_image.resize((new_W, new_H), Image.LANCZOS)
    else:
        input_image = original_image

    # 4. PREPARE TENSORS
    # Normalize to [-1, 1] for the model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    x = transform(input_image).unsqueeze(0).to(device)
    x.requires_grad_(True)
    
    # 5. DEFINE TARGET (The "Void")
    # Instead of a fixed 512x512 target, we generate a target matching the image size
    target_tensor = torch.zeros_like(x).to(device)
    with torch.no_grad():
        target_latents = vae.encode(target_tensor).latent_dist.mean

    # 6. PGD ATTACK (High-Res)
    original_x = x.clone().detach()
    alpha = 0.01 
    
    print(f"Attacking Image Resolution: {new_W}x{new_H}")
    
    for i in range(steps):
        # Forward Pass
        latents = vae.encode(x).latent_dist.mean
        
        # Loss: Make image look like "zeros" to the AI
        loss = torch.nn.functional.mse_loss(latents, target_latents)
        
        # Backward Pass
        grad = torch.autograd.grad(loss, x)[0]
        
        # Update Image
        x = x - alpha * grad.sign()
        
        # Project (Epsilon Constraint)
        delta = torch.clamp(x - original_x, min=-epsilon, max=epsilon)
        x = torch.clamp(original_x + delta, min=-1, max=1)
        
        # Reset Gradients
        x = x.detach()
        x.requires_grad_(True)
        
    # 7. RECOVER & RESTORE EXACT SIZE
    x = x.detach().cpu()
    x = (x / 2 + 0.5).clamp(0, 1) # Denormalize to [0, 1]
    
    img_array = x[0].permute(1, 2, 0).numpy()
    protected_image = Image.fromarray((img_array * 255).astype(np.uint8))
    
    # If we changed dimensions slightly for the VAE, change them back EXACTLY
    if protected_image.size != (W, H):
        protected_image = protected_image.resize((W, H), Image.LANCZOS)
    
    # Return as Bytes
    img_byte_arr = io.BytesIO()
    protected_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return img_byte_arr