# NeuroShield 
**Live at : https://fuego-alpha.vercel.app/**

**NeuroShield**  is a tool to immunize images against unauthorized AI manipulation using adversarial perturbations.

This project cross-platform supported (Local & Google Colab).

## Inspiration

We are living in an era where our digital photos can be stolen and manipulated without consent. A recent [BBC report](https://www.bbc.com/news/articles/cg7y10xm4x2o) highlighted a terrifying trend: students and everyday individuals are increasingly becoming victims of non-consensual deepfake pornography. This isn't just a celebrity issue anymore; it's a societal crisis.

While legislation moves slowly, technology moves fast. We were inspired by **MIT's PhotoGuard**, a research project that introduced the concept of using "adversarial perturbations" to disrupt AI models. We wanted to take that research out of the lab and build a practical, accessible web tool that empowers anyone to protect their photos before uploading them online.

## What it does

**Neuro Shield** is a web-based privacy tool that "immunizes" images against generative AI.

* **Invisible Cloaking:** Users upload a photo, and our engine applies a calculated layer of adversarial noise. To the human eye, the image looks unchanged. To an AI, the image content becomes unrecognizable.
* **Dual-Tier Protection:**
* **Simple Shield:** A lightweight defense against basic scraping.
* **Extreme Shield (Pro):** A heavy-duty, iterative gradient attack designed to break advanced Diffusion models.
* **So when an AI model tries to modify the image , either the model fails to do so or the edited image looks awfully fake.** 

* **Real-Time Verification:** Unlike other tools, we don't just ask you to trust us. We include a built-in **"Attack Simulation"**. Users can instantly challenge a Stable Diffusion model to edit their protected image, proving the shield's effectiveness in real-time.

## How we built it

We built Neuro Shield using a client-server architecture powered by **Google Colab** to leverage free GPU compute.

* **Backend:** We used **Python** and **FastAPI** to build the REST API. We utilized `pyngrok` to tunnel the localhost server from Colab to the public web.
* **AI Engine:** We utilized **Hugging Face Diffusers** and **PyTorch**.
* For the **Simple Shield**, we target the VAE (Variational Autoencoder) latent space.
* For the **Extreme Shield**, we perform **Projected Gradient Descent (PGD)** directly against the Stable Diffusion Inpainting UNet.
* For the **Deplyment ** here is the google Colab link [link](https://colab.research.google.com/drive/1WlabR7g8IefdW6KfGtvzL9m8v44tqA09?usp=sharing)


* **Frontend:** We built a custom HTML/CSS interface with a cyberpunk aesthetic to emphasize privacy and security, using vanilla JavaScript for asynchronous API communication.

## Challenges we ran into

* **The "Channel Mismatch" Nightmare:** When building the Verification feature, we tried to reuse the same model we used for protection. However, the Inpainting model expects 9 input channels (image + mask + masked_image), while standard image-generation tasks only provide 4. This caused `RuntimeError` crashes. We solved this by implementing a **Hybrid Architecture** that loads a standard UNet specifically for verification while keeping the Inpainting UNet for protection.
* **GPU Memory (VRAM) Constraints:** Running two distinct Diffusion pipelines + a VAE on a single free-tier NVIDIA T4 (16GB VRAM) led to immediate Out-Of-Memory crashes. We had to optimize aggressively by using `torch.float16`, manual garbage collection (`gc.collect()`), and a **Shared Memory Strategy** where both pipelines share the same Text Encoder and VAE components, saving ~3GB of VRAM.

## Accomplishments that we're proud of

* **Zero-Cost Infrastructure:** We successfully deployed a production-grade AI application with zero hosting costs by leveraging Google Colab and Ngrok.
* **The "Hybrid" Memory Architecture:** Solving the VRAM bottleneck allowed us to offer *both* heavy-duty protection and real-time verification in the same session, which is technically difficult on limited hardware.
* **Visual Proof:** Seeing the "Verification" result for the first time—where the AI tried to edit a face and produced gray static instead—was a huge moment of validation.

## What we learned

We learned that AI models see the world very differently than humans. We dove deep into the math of **Adversarial Machine Learning**. Specifically, we learned how to maximize the loss function  with respect to the input pixels  rather than the model weights :

We gained a practical understanding of how **Latent Diffusion Models** extract features and how fragile that extraction process really is when introduced to specific patterns of Gaussian noise.

## What's next for Neuro Shield

* **Scalable Cloud Deployment:** Moving from Google Colab to a dedicated GPU cluster (AWS SageMaker or RunPod) to remove the session timeout restrictions.
* **Multi-User Queueing:** Implementing Redis/Celery to handle thousands of concurrent users without blocking the GPU.
* **Browser-Side Inference:** Exploring WebGPU to run the "Simple Shield" directly in the user's browser, enhancing privacy by ensuring the original photo never leaves their device.

## Project Structure

-   `backend/`: Contains the Python FastAPI server and core logic.
    -   `core/`: Model loading, configuration, and attack algorithms.
    -   `api/`: API endpoint definitions.
-   `frontend/`: Contains the web interface.
    -   `index.html`: Main UI.
    -   `js/`: Application logic.
    -   `css/`: Styling.

---



## 🚀 How to Run

### Option 1: Google Colab (Recommended for GPU)

Since this project requires a GPU for the diffusion models, Google Colab is the easiest way to run the backend.

1.  **Clone the Repository** in a Colab cell:
    ```python
    !git clone https://github.com/lazyserp/NeuroShield.git
    %cd NeuroShield
    ```

2.  **Install Dependencies**:
    The backend handles most installs, but running this ensures everything is ready:
    ```python
    !pip install fastapi uvicorn python-multipart pyngrok nest_asyncio torch diffusers transformers accelerate scipy ftfy
    ```

3.  **Start the Server**:
    Run the following in a cell. **Make sure to set your Ngrok Token.**
    
    ```python
    import os
    
    # 1. Set your Ngrok Token (Get one from https://dashboard.ngrok.com/get-started/your-authtoken)
    os.environ["NGROK_TOKEN"] = "YOUR_NGROK_TOKEN_HERE"
    
    # 2. Run the Server
    # We use !python -m to run it as a module
    !python -m backend.server
    ```
    
    *Expected Output:* You will see a `🔗 SERVER URL: https://....ngrok-free.app` in the output. Copy this URL.

4.  **Connect Frontend**:
    -   Open the `frontend/index.html` file (either locally or deployed on Vercel).
    -   Paste the Ngrok URL into the "Server URL" box in the web interface.
    -   Upload an image and start immunizing!

---

### Option 2: Local Machine (Requires NVIDIA GPU)

1.  **Prerequisites**:
    -   Python 3.10+
    -   CUDA-capable GPU (at least 6GB VRAM recommended).

2.  **Setup**:
    ```bash
    # Clone the repo
    git clone https://github.com/YOUR_USERNAME/NeuroShield.git
    cd NeuroShield
    
    # Install dependencies
    pip install -r backend/requirements.txt
    # OR let the script install them automatically on first run
    ```

3.  **Run**:
    Set your ngrok token (optional, only if you need public access, otherwise localhost works for specific setups, but the frontend defaults to checking a URL).
    
    ```bash
    # CMD / PowerShell
    set NGROK_TOKEN=your_token_here
    python -m backend.server
    ```
    
    *Note: Using -m backend.server is crucial.*

4.  **Frontend**:
    -   Simply open `frontend/index.html` in your browser.

---

## Deployment (Vercel)

For the frontend:
-   Connect your GitHub repo to Vercel.
-   Set the **Root Directory** to `frontend/`.
-   No build command is needed (it's static HTML/JS).
