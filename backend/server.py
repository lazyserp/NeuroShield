import os
import uvicorn
import nest_asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok

# Import endpoints
from backend.api.endpoints import router as api_router

# --- CONFIG & AUTH ---
# Doing this before app startup
token = os.environ.get("NGROK_TOKEN")
if not token:
    # Check if we can get it from input, usually tough in detached mode, but preserving logic
    print("Warning: NGROK_TOKEN not set in environment.") 
    # Logic to ask for input removed for server file stability, user should set env var
    # or rely on hardcoded auth if they had it. 
    # For now, we follow original:
    # token = input("Paste Ngrok Token: ") 
    # We will skip input() to avoid hanging if run non-interactively.
    pass

if token:
    ngrok.set_auth_token(token)

nest_asyncio.apply()

# --- APP SETUP ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

app.include_router(api_router)

# --- STARTUP ---
if __name__ == "__main__":
    ngrok.kill()
    try:
        public_url = ngrok.connect(8000).public_url
        print(f"\n🔗 SERVER URL: {public_url}")
        print("👉 Update your Frontend with this new URL")
    except Exception as e:
        print(f"Ngrok Error: {e}")

    # Use port 8000 as per original
    uvicorn.run(app, host="0.0.0.0", port=8000)
