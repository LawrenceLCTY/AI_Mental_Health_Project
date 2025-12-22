# AI Mental Health Content Generator

Purpose: generate safe, non-diagnostic mental-health posts (text via Qwen models) and optional images (Stable Diffusion XL).

### Setup

```bash
# create env and install deps
conda create -n aimh python=3.10 -y
conda activate aimh
pip install -r requirements.txt
```

If you have a GPU, install a matching `torch` wheel per your CUDA version before running.

### Models

Download models to local repository
```bash
mkdir models
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir ./models/Qwen/Qwen3-VL-8B-Instruct
hf download stabilityai/stable-diffusion-xl-base-1.0 --local-dir ./models/stabilityai/stable-diffusion-xl-base-1.0
```

.env (example)
```
QWEN_VL=./models/Qwen/Qwen3-VL-8B-Instruct
SD_MODEL=./models/stabilityai/stable-diffusion-xl-base-1.0
```

### Run backend
```bash
python app.py      # starts uvicorn on port 8004
# or
uvicorn app:app --host 0.0.0.0 --port 8004 --reload
```

### Run fontend
```bash
python ui.py       # Gradio UI on port 7860 (talks to backend at 127.0.0.1:8004)
```

### Outputs
- Images: `outputs/images/` (served at `/outputs/images/...`)

### Quick troubleshooting
- If imports fail, install a compatible `torch` for your platform/GPU.
- If SDXL isn't available, `/image` will skip rendering and log a warning.
- Simple safety checks live in `utils/safety.py` and run as a FastAPI dependency.
