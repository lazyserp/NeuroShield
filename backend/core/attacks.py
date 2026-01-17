import torch
import torch.nn.functional as F
from .config import device

# --- ATTACK ALGORITHMS ---

def attack_simple(X, model, eps=0.05, steps=40):
    delta = torch.zeros_like(X).uniform_(-eps, eps).to(device)
    delta.requires_grad = True
    step_size = 0.01
    
    for _ in range(steps):
        adv_image = X + delta
        # model here is vae.encode passed from caller
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
        # dtype match
        mask = torch.zeros((batch_size, 1, latents.shape[2], latents.shape[3]), device=device, dtype=latents.dtype)
        masked_latents = latents # If mask is 0, masked_latents is just latents
        
        # Concatenate inputs as Inpainting UNet expects 9 channels
        latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)
        
        # Pass encoder_hidden_states as dummy
        dummy_encoder_hidden_states = torch.zeros((1, 77, 768), device=device).half()
        
        noise_pred = pipe.unet(latent_model_input, timesteps, encoder_hidden_states=dummy_encoder_hidden_states)[0]
        
        loss = (X_adv - target_image).norm() 
        grad = torch.autograd.grad(loss, [X_adv])[0]
        
        X_adv = X_adv - step_size * grad.sign()
        X_adv = torch.minimum(torch.maximum(X_adv, image_tensor - eps), image_tensor + eps)
        X_adv.data = torch.clamp(X_adv, min=-1, max=1)
        
        if i % 5 == 0: torch.cuda.empty_cache()
            
    return X_adv
