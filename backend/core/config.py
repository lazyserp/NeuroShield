import torch
import sys
import subprocess
import os

# --- CONFIG & UTILS ---

def install_packages():
    print("⏳ Checking libraries...")
    packages = "fastapi uvicorn python-multipart pyngrok nest_asyncio torch diffusers transformers accelerate scipy ftfy"
    subprocess.check_call([sys.executable, "-m", "pip", "install", *packages.split()])
    print("✅ Libraries Installed.")

# Check imports
try:
    import pyngrok
    import diffusers
    import fastapi
except ImportError:
    install_packages()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Engine Starting on {device}...")
