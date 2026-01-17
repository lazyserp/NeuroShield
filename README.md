# NeuroShield 🛡️

**NeuroShield** (formerly PhotoGuard/Replication) is a tool to immunize images against unauthorized AI manipulation using adversarial perturbations.

This project cross-platform supported (Local & Google Colab).

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
