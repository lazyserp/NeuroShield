// NeuroShield Frontend Logic

let currentMode = 'simple';
let protectedBlob = null;

const fileInput = document.getElementById('fileInput');
const apiUrlInput = document.getElementById('apiUrl');
const immunizeBtn = document.getElementById('immunizeBtn');
const statusText = document.getElementById('statusText');
const dropZone = document.getElementById('dropZone');

// Shield Selection Logic
function selectShield(mode) {
    currentMode = mode;
    document.querySelectorAll('.shield-option').forEach(el => el.classList.remove('selected'));
    if (mode === 'simple') {
        document.getElementById('btnSimple').classList.add('selected');
        if (immunizeBtn) {
            immunizeBtn.innerText = "Initialize Protection (Free)";
            immunizeBtn.classList.remove('btn-pro');
        }
    } else {
        document.getElementById('btnExtreme').classList.add('selected');
        if (immunizeBtn) {
            immunizeBtn.innerText = "Initialize Extreme Shield";
            immunizeBtn.classList.add('btn-pro');
        }
    }
}

// Event Listeners
if (dropZone && fileInput) {
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.borderColor = '#fb64b6'; });
    dropZone.addEventListener('dragleave', (e) => { e.preventDefault(); dropZone.style.borderColor = '#404040'; });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#404040';
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
}

function handleFile(file) {
    if (file) {
        const preview = document.getElementById('originalPreview');
        if (preview) preview.src = URL.createObjectURL(file);

        document.getElementById('previewContainer').classList.remove('hidden');
        dropZone.style.display = 'none';
        if (immunizeBtn) immunizeBtn.disabled = false;
    }
}

if (apiUrlInput) {
    apiUrlInput.addEventListener('input', async () => {
        const url = apiUrlInput.value.replace(/\/$/, "");
        try {
            const res = await fetch(`${url}/`);
            if (res.ok) {
                document.getElementById('connectionStatus').style.backgroundColor = "#4ade80";
                document.getElementById('connectionStatus').style.boxShadow = "0 0 10px #4ade80";
                document.getElementById('connectionStatus').title = "Online";
            }
        } catch (e) {
            document.getElementById('connectionStatus').style.backgroundColor = "#ef4444";
            document.getElementById('connectionStatus').style.boxShadow = "0 0 10px #ef4444";
        }
    });
}

// Main Actions (Expose to window for onclick handlers in HTML)
window.processImage = async function () {
    const file = fileInput.files[0];
    let baseUrl = apiUrlInput.value.replace(/\/$/, "");
    if (!baseUrl || !file) return alert("Check Server URL and File.");

    immunizeBtn.disabled = true;
    document.getElementById('status').classList.remove('hidden');
    document.getElementById('resultSection').classList.add('hidden');
    document.getElementById('verifySection').classList.add('hidden');

    statusText.innerText = currentMode === 'extreme'
        ? "Running Diffusion Gradient Attack..."
        : "Calculating Adversarial Noise...";

    const formData = new FormData();
    formData.append("file", file);
    formData.append("mode", currentMode);

    try {
        const response = await fetch(`${baseUrl}/immunize`, {
            method: "POST",
            body: formData
        });
        if (!response.ok) throw new Error("Processing Failed");

        protectedBlob = await response.blob();
        const url = URL.createObjectURL(protectedBlob);

        document.getElementById('protectedPreview').src = url;
        document.getElementById('verifyInputImg').src = url;
        document.getElementById('downloadLink').href = url;

        document.getElementById('status').classList.add('hidden');
        document.getElementById('resultSection').classList.remove('hidden');
        document.getElementById('verifySection').classList.remove('hidden');
        immunizeBtn.disabled = false;

    } catch (error) {
        alert("Error: " + error.message);
        document.getElementById('status').classList.add('hidden');
        immunizeBtn.disabled = false;
    }
}

window.runVerification = async function () {
    if (!protectedBlob) return alert("No protected image found!");
    let baseUrl = apiUrlInput.value.replace(/\/$/, "");
    const prompt = document.getElementById('verifyPrompt').value;

    document.getElementById('verifyBtn').disabled = true;
    document.getElementById('verifyStatus').classList.remove('hidden');
    document.getElementById('verifyResult').classList.add('hidden');

    const formData = new FormData();
    formData.append("file", protectedBlob, "protected.png");
    formData.append("prompt", prompt);

    try {
        const response = await fetch(`${baseUrl}/verify`, {
            method: "POST",
            body: formData
        });
        if (!response.ok) throw new Error("Verification Failed");

        const blob = await response.blob();
        document.getElementById('verifyOutputImg').src = URL.createObjectURL(blob);

        document.getElementById('verifyStatus').classList.add('hidden');
        document.getElementById('verifyResult').classList.remove('hidden');
        document.getElementById('verifyBtn').disabled = false;

    } catch (error) {
        alert("Verification Error: " + error.message);
        document.getElementById('verifyStatus').classList.add('hidden');
        document.getElementById('verifyBtn').disabled = false;
    }
}

// Initial binding for shield selection (since they have onclick in HTML)
window.selectShield = selectShield;
