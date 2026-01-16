from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from attack import immunize_image

app = FastAPI()

# Allow the frontend to talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "NeuroShield System Online"}

@app.post("/immunize")
async def process_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    # Run the Attack
    protected_img_io = immunize_image(image_bytes)
    
    return StreamingResponse(protected_img_io, media_type="image/png")