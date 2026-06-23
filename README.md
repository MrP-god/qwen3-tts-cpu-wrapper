# Qwen3-TTS CPU Verber & Gradio Studio

An optimized, CPU-friendly API wrapper ("Verber") and interactive **Gradio** studio for Alibaba's **Qwen3-TTS** text-to-speech models. Built with Astral `uv`, this project runs entirely on AMD/Intel CPUs on Windows without requiring CUDA.

It serves a dual purpose:
1.  **Programmatic Integration:** Exposes clean REST API endpoints for your *other projects* to send text and receive real-time, low-latency WAV file streams.
2.  **Voice Design Studio:** Launches a beautiful, interactive Gradio Web UI at the root (`/`) to test, design, and clone voices when not running automation.

---

## 🚀 Architectural Design

This wrapper combines a **FastAPI** REST backend with a **Gradio** frontend running in the same process:

```
                  ┌──────────────────────────────┐
                  │      FastAPI App (8000)      │
                  └──────────────┬───────────────┘
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
┌──────────────────┐                            ┌──────────────────┐
│   REST API       │                            │   Gradio UI      │
│  (For other      │                            │  (For voice      │
│   projects)      │                            │   designing)     │
└────────┬─────────┘                            └────────┬─────────┘
         │                                               │
         └───────────────┬───────────────────────────────┘
                         ▼
             ┌───────────────────────┐
             │     ModelManager      │
             │  (Async CPU Threads)  │
             └───────────┬───────────┘
                         ▼
             ┌───────────────────────┐
             │      Qwen3-TTS        │
             │   (0.6B / 1.7B)       │
             └───────────────────────┘
```

---

## 🛠️ Windows Installation (AMD/Intel CPU)

The installer utilizes Astral **`uv`** for environment setup and package management, which is significantly faster and more reliable than standard pip.

1.  Clone or download this repository to your Windows machine.
2.  Ensure you have **Python 3.12** installed and added to your PATH.
3.  Double-click **`run.bat`**. This script will:
    *   Check for `uv` (and automatically install it if missing).
    *   Create a virtual environment (`.venv`) using `uv venv`.
    *   Install the optimized CPU-only PyTorch build using `uv pip`.
    *   Install `qwen-tts`, `gradio`, and remaining dependencies.
    *   Launch the server on `http://127.0.0.1:8000`.
4.  Open your browser and navigate to **`http://localhost:8000`** to access the Gradio Studio.

---

## 🍏 macOS / Linux Installation (Development)

To run the wrapper on your development Mac:

1.  Ensure the shell launcher is executable:
    ```bash
    chmod +x run.sh
    ```
2.  Execute the shell script:
    ```bash
    ./run.sh
    ```
3.  Navigate to `http://localhost:8000`.

---

## 🔌 Programmatic Integration (For Your Other Project)

Your other project can fetch voice conversions in real-time by sending standard HTTP requests to the FastAPI backend.

### 1. Synthesize Speech (Custom Voice or Voice Design)
Send text to have it converted into voice bytes instantly.

*   **Endpoint:** `POST /synthesize`
*   **Request JSON:**
    ```json
    {
      "text": "Hello! This text is converted in real-time.",
      "language": "English",
      "speaker": "aiden",
      "instruct": "speak in an excited, rapid tone",
      "streaming": true
    }
    ```
    *Note: Set `"streaming": true` to enable low-latency internal generation path.*
*   **Response:** Binary audio stream (`audio/wav`)

#### Example Integration (Python / requests)
```python
import requests

url = "http://127.0.0.1:8000/synthesize"
payload = {
    "text": "The quick brown fox jumps over the lazy dog.",
    "language": "English",
    "speaker": "vivian",
    "instruct": "confident, clear voice",
    "streaming": true
}

response = requests.post(url, json=payload)
if response.status_code == 200:
    with open("output.wav", "wb") as f:
        f.write(response.content)
    print("Voice conversion successful. Audio saved to output.wav.")
else:
    print("Error:", response.json())
```

#### Example Integration (JavaScript / Fetch)
```javascript
const response = await fetch("http://127.0.0.1:8000/synthesize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
        text: "Real-time speech conversion.",
        language: "English",
        speaker: "ryan",
        streaming: true
    })
});

if (response.ok) {
    const audioBlob = await response.blob();
    const audioUrl = URL.createObjectURL(audioBlob);
    // You can now feed this URL to an HTML5 Audio player
    const audio = new Audio(audioUrl);
    audio.play();
}
```

---

## 👥 Voice Cloning Endpoint
If your other project needs to clone a voice on the fly, upload the reference file via form-data.

*   **Endpoint:** `POST /clone`
*   **Request Body (Multipart Form-Data):**
    *   `text` (string): Target text to speak.
    *   `language` (string, default `"English"`): Target language.
    *   `ref_audio` (binary file): WAV or MP3 recording of reference speaker (~3-10s).
    *   `ref_text` (string, optional): Transcript of the reference audio clip.
*   **Response:** Binary audio stream (`audio/wav`)

---

## ⚡ Troubleshooting & Performance Tips

*   **CPU Overhead:** Ensure you are using the **0.6B** checkpoints (`0.6B-CustomVoice` or `0.6B-Base`) for real-time speed.
*   **Data Type:** If your CPU supports AVX-512 vector arithmetic, switch the precision dropdown to **`bfloat16`** to decrease processing times.
*   **Timeout Prevention:** Heavy audio generations run on asynchronous thread pools, guaranteeing that programmatic HTTP connections do not hang or drop.
