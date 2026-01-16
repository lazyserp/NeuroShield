import torch
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms
import numpy as np
import io

# Setup Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"NeuroShield Loading on: {device}")

# Load Model (Global variable to load only once)
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device)
vae.requires_grad_(False)

def immunize_image(image_bytes, epsilon=0.03, steps=50):
    # 1. Prepare Image
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((512, 512))
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    x = transform(input_image).unsqueeze(0).to(device)
    x.requires_grad_(True)
    
    # 2. Target (Grey Void)
    target_tensor = torch.zeros_like(x).to(device)
    with torch.no_grad():
        target_latents = vae.encode(target_tensor).latent_dist.mean

    # 3. PGD Attack
    original_x = x.clone().detach()
    alpha = 0.01
    
    for i in range(steps):
        latents = vae.encode(x).latent_dist.mean
        loss = torch.nn.functional.mse_loss(latents, target_latents)
        grad = torch.autograd.grad(loss, x)[0]
        
        x = x - alpha * grad.sign()
        delta = torch.clamp(x - original_x, min=-epsilon, max=epsilon)
        x = torch.clamp(original_x + delta, min=-1, max=1)
        
        x = x.detach()
        x.requires_grad_(True)
        
    # 4. Process Output
    x = x.detach().cpu()
    x = (x / 2 + 0.5).clamp(0, 1)
    img_array = x[0].permute(1, 2, 0).numpy()
    final_image = Image.fromarray((img_array * 255).astype(np.uint8))
    
    # Return as Bytes
    img_byte_arr = io.BytesIO()
    final_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr