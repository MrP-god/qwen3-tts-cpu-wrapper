import os
import io
import asyncio
import threading
import tempfile
import psutil
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import torch
import soundfile as sf
import numpy as np
import gradio as gr

# Try importing Qwen3TTSModel
try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    Qwen3TTSModel = None

# Initialize FastAPI App
app = FastAPI(
    title="Qwen3-TTS CPU Verber",
    description="A CPU-friendly FastAPI wrapper and Web UI for Qwen3-TTS models.",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model Manager to handle loading/unloading Qwen3-TTS checkpoints
class ModelManager:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.device = "cpu"
        self.dtype_str = "float32"
        self.is_loading = False
        self.error = None
        self.lock = threading.Lock()

    def get_status(self):
        with self.lock:
            # Calculate memory usage
            process = psutil.Process(os.getpid())
            ram_mb = process.memory_info().rss / (1024 * 1024)
            
            return {
                "model_name": self.model_name,
                "device": self.device,
                "dtype": self.dtype_str,
                "is_loaded": self.model is not None,
                "is_loading": self.is_loading,
                "error": self.error,
                "ram_usage_mb": round(ram_mb, 2)
            }

    def load(self, model_name: str, device: str = "cpu", dtype_str: str = "float32"):
        with self.lock:
            if self.is_loading:
                raise RuntimeError("Model is already loading in the background.")
            self.is_loading = True
            self.error = None

        def _target():
            try:
                # Set torch dtype
                if dtype_str == "bfloat16":
                    dtype = torch.bfloat16
                elif dtype_str == "float16":
                    dtype = torch.float16
                else:
                    dtype = torch.float32

                print(f"[ModelManager] Loading {model_name} on {device} with {dtype_str}...")
                
                if Qwen3TTSModel is None:
                    raise ImportError("The 'qwen-tts' package is not installed.")

                # Load model
                model = Qwen3TTSModel.from_pretrained(
                    model_name,
                    device_map=device,
                    dtype=dtype
                )
                
                with self.lock:
                    self.model = model
                    self.model_name = model_name
                    self.device = device
                    self.dtype_str = dtype_str
                    self.error = None
                print(f"[ModelManager] Successfully loaded {model_name}.")
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"[ModelManager] Error loading model: {e}\n{tb}")
                with self.lock:
                    self.model = None
                    self.model_name = None
                    self.error = f"{str(e)}\n{tb}"
            finally:
                with self.lock:
                    self.is_loading = False

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()

model_manager = ModelManager()

# Start loading the default 0.6B CustomVoice model on startup
@app.on_event("startup")
def startup_event():
    default_model = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    try:
        model_manager.load(default_model, device="cpu", dtype_str="float32")
    except Exception as e:
        print(f"Failed to initiate startup model loading: {e}")

class LoadRequest(BaseModel):
    model_name: str
    device: str = "cpu"
    dtype: str = "float32"

class SynthesizeRequest(BaseModel):
    text: str
    language: str = "English"
    speaker: str = "default"
    instruct: Optional[str] = None
    streaming: bool = False

@app.get("/status")
def get_status():
    return model_manager.get_status()

@app.post("/load")
def load_model(req: LoadRequest):
    try:
        model_manager.load(req.model_name, device=req.device, dtype_str=req.dtype)
        return {"status": "loading", "message": f"Started loading {req.model_name} in the background."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/speakers")
def get_speakers():
    status = model_manager.get_status()
    if not status["is_loaded"]:
        return {"speakers": ["default", "aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"]}
    
    model = model_manager.model
    if hasattr(model, "get_supported_speakers"):
        try:
            return {"speakers": model.get_supported_speakers()}
        except Exception as e:
            print(f"Error calling get_supported_speakers: {e}")
    
    return {"speakers": ["default", "aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"]}

@app.get("/languages")
def get_languages():
    return {
        "languages": [
            "English", "Chinese", "Japanese", "Korean", "German",
            "French", "Russian", "Portuguese", "Spanish", "Italian"
        ]
    }

@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    status = model_manager.get_status()
    if not status["is_loaded"]:
        raise HTTPException(status_code=400, detail="No model is loaded yet.")
    
    model = model_manager.model
    if not hasattr(model, "generate_custom_voice"):
        raise HTTPException(
            status_code=400, 
            detail="The loaded model does not support preset voices or design-based synthesis."
        )

    try:
        print(f"Synthesizing: '{req.text}' using {status['model_name']}...")
        
        # Run inference in a background thread
        wavs, sr = await asyncio.to_thread(
            model.generate_custom_voice,
            text=req.text,
            language=req.language,
            speaker=req.speaker,
            instruct=req.instruct,
            non_streaming_mode=not req.streaming
        )

        audio_data = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sr, format="WAV")
        buffer.seek(0)
        
        return StreamingResponse(
            buffer, 
            media_type="audio/wav", 
            headers={"Content-Disposition": "attachment; filename=synthesized.wav"}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

@app.post("/clone")
async def clone(
    text: str = Form(...),
    language: str = Form("English"),
    ref_text: Optional[str] = Form(None),
    ref_audio: UploadFile = File(...)
):
    status = model_manager.get_status()
    if not status["is_loaded"]:
        raise HTTPException(status_code=400, detail="No model is loaded yet.")
    
    model = model_manager.model
    if not hasattr(model, "generate_voice_clone"):
        raise HTTPException(
            status_code=400, 
            detail="The loaded model does not support voice cloning."
        )

    suffix = os.path.splitext(ref_audio.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = temp_file.name
        content = await ref_audio.read()
        temp_file.write(content)

    try:
        print(f"Cloning voice for: '{text}' using {status['model_name']}...")
        
        wavs, sr = await asyncio.to_thread(
            model.generate_voice_clone,
            text=text,
            language=language,
            ref_audio=temp_path,
            ref_text=ref_text
        )

        audio_data = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sr, format="WAV")
        buffer.seek(0)
        
        return StreamingResponse(
            buffer, 
            media_type="audio/wav", 
            headers={"Content-Disposition": "attachment; filename=cloned.wav"}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Cloning failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ==========================================
# GRADIO WEB INTERFACE
# ==========================================

languages_list = ["English", "Chinese", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"]

def get_status_md():
    status = model_manager.get_status()
    is_loaded_str = "🟢 **Ready**" if status["is_loaded"] else ("🟡 **Loading Checkpoint...**" if status["is_loading"] else "🔴 **No Model Loaded**")
    err_str = f"\n\n⚠️ **Error:** {status['error']}" if status["error"] else ""
    return f"""
### 💻 System Status Monitor
* **Loaded Checkpoint:** `{status['model_name'] or 'None'}`
* **Engine Status:** {is_loaded_str}
* **Hardware Device:** `{status['device'].toUpperCase() if hasattr(status['device'], 'toUpperCase') else status['device']}`
* **Precision Format:** `{status['dtype']}`
* **Memory RSS Allocation:** `{status['ram_usage_mb']} MB`{err_str}
"""

def gradio_load_model(model_name, dtype):
    try:
        model_manager.load(model_name, device="cpu", dtype_str=dtype)
        return "🔄 Switch triggered! Please monitor status block below as the model loads asynchronously."
    except Exception as e:
        return f"❌ Failed to trigger load: {str(e)}"

def gradio_synthesize(text, language, speaker, instruct, streaming):
    status = model_manager.get_status()
    if not status["is_loaded"]:
        return None, "❌ No model loaded. Please load a checkpoint first."
    model = model_manager.model
    if not hasattr(model, "generate_custom_voice"):
        return None, "❌ The loaded model is not compatible with Preset Voices. Load a CustomVoice or VoiceDesign checkpoint."
    
    try:
        # Run inference
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct if instruct else None,
            non_streaming_mode=not streaming
        )
        audio_data = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        
        # Save temp file
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_wav.name, audio_data, sr)
        return temp_wav.name, None
    except Exception as e:
        return None, f"❌ Generation error: {str(e)}"

def gradio_design(text, language, instruct, streaming):
    status = model_manager.get_status()
    if not status["is_loaded"]:
        return None, "❌ No model loaded. Please load a checkpoint first."
    model = model_manager.model
    if not hasattr(model, "generate_custom_voice"):
        return None, "❌ The loaded model is not compatible with Voice Design. Load a VoiceDesign checkpoint."
    
    try:
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=language,
            speaker="default",
            instruct=instruct,
            non_streaming_mode=not streaming
        )
        audio_data = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_wav.name, audio_data, sr)
        return temp_wav.name, None
    except Exception as e:
        return None, f"❌ Design error: {str(e)}"

def gradio_clone(text, language, ref_audio_path, ref_text):
    status = model_manager.get_status()
    if not status["is_loaded"]:
        return None, "❌ No model loaded. Please load a checkpoint first."
    model = model_manager.model
    if not hasattr(model, "generate_voice_clone"):
        return None, "❌ The loaded model does not support voice cloning. Load a Base checkpoint."
    if not ref_audio_path:
        return None, "❌ Please record or upload a reference audio file first."
        
    try:
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio_path,
            ref_text=ref_text if ref_text else None
        )
        audio_data = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_wav.name, audio_data, sr)
        return temp_wav.name, None
    except Exception as e:
        return None, f"❌ Cloning error: {str(e)}"

def update_gradio_speakers():
    speakers = get_speakers()["speakers"]
    return gr.update(choices=speakers, value=speakers[0] if speakers else "default")


# Build Gradio UI Blocks
with gr.Blocks(title="Qwen3-TTS CPU Verber", theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="teal", neutral_hue="slate")) as demo:
    gr.Markdown("# 🔊 Qwen3-TTS CPU Verber Studio")
    gr.Markdown("An optimized, CUDA-free interface for text-to-speech, custom voice design, and zero-shot voice cloning.")
    
    with gr.Row():
        # Left column: Model manager & status monitor
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Checkpoint Settings")
            model_dd = gr.Dropdown(
                label="Qwen3-TTS Model Checkpoint",
                choices=[
                    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                    "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
                ],
                value="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                info="0.6B is recommended for fast CPU-only execution."
            )
            dtype_dd = gr.Dropdown(
                label="Precision Format",
                choices=["float32", "bfloat16"],
                value="float32",
                info="float32 is highly compatible; bfloat16 matches modern CPU vector math."
            )
            btn_load = gr.Button("🔄 Switch Checkpoint", variant="secondary")
            load_out = gr.Textbox(label="Action Log", placeholder="No action triggered.")
            
            # Auto-updating status
            status_box = gr.Markdown(get_status_md())
            status_timer = gr.Timer(3.0)
            status_timer.tick(get_status_md, outputs=status_box)

        # Right column: Workspace Tabs
        with gr.Column(scale=2):
            with gr.Tabs():
                # Tab 1: Preset Custom Voices
                with gr.Tab("Preset Custom Voice"):
                    gr.Markdown("### Synthesize Speech with Curated Timbres")
                    with gr.Row():
                        speaker_dd = gr.Dropdown(label="Speaker Timbre", choices=["default", "aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"], value="default")
                        lang_dd = gr.Dropdown(label="Language", choices=languages_list, value="English")
                    
                    instruct_tb = gr.Textbox(label="Style / Emotional Prompt (Optional)", placeholder="e.g. Speak in a sad, slow, crying voice")
                    text_tb = gr.Textbox(label="Speech Input Text", lines=4, placeholder="Enter text you want the preset voice to say...")
                    stream_cb = gr.Checkbox(label="Enable Low-Latency Optimization", value=False)
                    
                    btn_synth = gr.Button("⚡ Generate Speech", variant="primary")
                    audio_out = gr.Audio(label="Synthesized Audio", type="filepath")
                    err_out = gr.Textbox(label="Error Logs", visible=False)

                    # Interactivity
                    btn_synth.click(
                        gradio_synthesize, 
                        inputs=[text_tb, lang_dd, speaker_dd, instruct_tb, stream_cb],
                        outputs=[audio_out, err_out]
                    )

                # Tab 2: Voice Design
                with gr.Tab("Voice Design"):
                    gr.Markdown("### Create a Timbre from Natural Language Descriptions")
                    gr.Markdown("⚠️ *Works best with the **1.7B-VoiceDesign** checkpoint.*")
                    
                    lang_design_dd = gr.Dropdown(label="Language", choices=languages_list, value="English")
                    instruct_design_tb = gr.Textbox(label="Voice Timbre Description Prompt", placeholder="e.g. A mature female speaker with a warm, breathy voice, speaking with high energy.")
                    text_design_tb = gr.Textbox(label="Speech Input Text", lines=4, placeholder="Enter text you want the designed voice to speak...")
                    stream_design_cb = gr.Checkbox(label="Enable Low-Latency Optimization", value=False)
                    
                    btn_synth_design = gr.Button("✨ Design & Synthesize", variant="primary")
                    audio_design_out = gr.Audio(label="Designed Audio", type="filepath")
                    err_design_out = gr.Textbox(label="Error Logs", visible=False)

                    # Interactivity
                    btn_synth_design.click(
                        gradio_design,
                        inputs=[text_design_tb, lang_design_dd, instruct_design_tb, stream_design_cb],
                        outputs=[audio_design_out, err_design_out]
                    )

                # Tab 3: Voice Cloning
                with gr.Tab("Voice Cloning"):
                    gr.Markdown("### Zero-Shot Voice Cloning from Reference Clip")
                    gr.Markdown("⚠️ *Requires a **Base** checkpoint loaded.*")
                    
                    lang_clone_dd = gr.Dropdown(label="Target Language", choices=languages_list, value="English")
                    ref_audio_input = gr.Audio(label="Reference Audio Clip (approx. 3-10s)", type="filepath", sources=["upload", "microphone"])
                    ref_text_tb = gr.Textbox(label="Reference Audio Transcription (Optional but recommended)", placeholder="Type the words spoken in the reference audio to assist cloning accuracy.")
                    text_clone_tb = gr.Textbox(label="New Text to Synthesize", lines=4, placeholder="Enter the new text you want the cloned voice to speak...")
                    
                    btn_synth_clone = gr.Button("👥 Clone & Synthesize", variant="primary")
                    audio_clone_out = gr.Audio(label="Cloned Audio Output", type="filepath")
                    err_clone_out = gr.Textbox(label="Error Logs", visible=False)

                    # Interactivity
                    btn_synth_clone.click(
                        gradio_clone,
                        inputs=[text_clone_tb, lang_clone_dd, ref_audio_input, ref_text_tb],
                        outputs=[audio_clone_out, err_clone_out]
                    )
                    
    # Setup actions
    btn_load.click(gradio_load_model, inputs=[model_dd, dtype_dd], outputs=[load_out])
    # When checkpoint changes, update the speaker list if CustomVoice model
    btn_load.click(update_gradio_speakers, outputs=[speaker_dd])

# Mount Gradio into FastAPI App
app = gr.mount_to_app(app, demo, path="/")
