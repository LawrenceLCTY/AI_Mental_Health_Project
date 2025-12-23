# AI Mental Health Content Generator (concise)

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
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir ./models/Qwen/Qwen3-VL-4B-Instruct
hf download stabilityai/stable-diffusion-xl-base-1.0 --local-dir ./models/stabilityai/stable-diffusion-xl-base-1.0
```

.env (example)
```
QWEN_VL=./models/Qwen/Qwen3-VL-8B-Instruct
SD_MODEL=./models/stabilityai/stable-diffusion-xl-base-1.0
GEMINI_API_KEY = YOUR_API_KEY
DEEPSEEK_API_KEY = YOUR_API_KEY
```

### Run
```bash
streamlit run app.py    
```

