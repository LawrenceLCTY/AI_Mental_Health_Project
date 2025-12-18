import os
import re
import json
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Request, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
from PIL import Image
import torch
import uuid
from datetime import datetime
import base64
from io import BytesIO

# Qwen3-VL specific imports
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# local helpers
import utils.safety as safety_utils
import utils.io_utils as io_utils

load_dotenv()
logging.basicConfig(level=logging.INFO)

# ---------- Config ----------
QWEN_VL = os.getenv("QWEN_VL")
SD_MODEL = os.getenv("SD_MODEL")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Device: {DEVICE}")

# Create output directories
OUTPUT_DIR = Path("outputs")
IMAGE_DIR = OUTPUT_DIR / "images"
UPLOAD_DIR = OUTPUT_DIR / "uploads"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Load Qwen3-VL with MULTIMODAL support ----------
logging.info(f"Loading Qwen3-VL from {QWEN_VL}...")

# Use the correct model class for multimodal
vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
    QWEN_VL,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    local_files_only=True,
    device_map="auto"  # Automatically handles multi-GPU if needed
)

processor = AutoProcessor.from_pretrained(
    QWEN_VL,
    trust_remote_code=True,
    local_files_only=True
)

logging.info(f"Qwen3-VL loaded successfully")

# ---------- Load SDXL ----------
sd_pipe = None
sd_tokenizer = None
try:
    sd_pipe = StableDiffusionXLPipeline.from_pretrained(
        SD_MODEL, 
        torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32,
        local_files_only=True,
        use_safetensors=True,
        variant="fp16" if DEVICE=="cuda" else None
    )
    sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
    sd_pipe = sd_pipe.to(DEVICE)
    sd_tokenizer = sd_pipe.tokenizer
    logging.info("SDXL Base loaded successfully")
except Exception as e:
    logging.warning(f"SDXL not loaded: {e}")

# ---------- FastAPI setup ----------
app = FastAPI(title="AIMH-Qwen3-VL Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# ---------- Request Models ----------
class IntentRequest(BaseModel):
    fragments: str
    user_profile: Optional[Dict[str, Any]] = None
    image_path: Optional[str] = None  # Optional image input

class OutlineRequest(BaseModel):
    intent: str
    slots: Dict[str, Any]
    image_path: Optional[str] = None  # Optional image context

class GenerateRequest(BaseModel):
    outline: List[str]
    style_examples: Optional[List[Dict[str,str]]] = None
    tone: Optional[str] = "温和且专业"
    length: Optional[str] = "短文"
    image_path: Optional[str] = None  # Optional image context

class CriticRequest(BaseModel):
    text: str
    require_safe: Optional[bool] = True

class ImageRequest(BaseModel):
    theme: str
    style: Optional[str] = "简洁插画, 温暖色调"
    composition: Optional[str] = "single character, calm setting, minimal background"

# ---------- Unified Text Generation (supports text + images) ----------
def generate_with_qwen3vl(
    prompt: str,
    image_path: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7
) -> str:
    """
    Generate text using Qwen3-VL
    Supports both text-only and text+image inputs
    """
    try:
        # Build message content
        content = []
        
        # Add image if provided
        if image_path:
            content.append({
                "type": "image",
                "image": image_path
            })
        
        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt
        })
        
        # Format as chat message
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        # Apply chat template
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process vision info if image present
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Prepare inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(vl_model.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = vl_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        # Trim input from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text.strip()
        
    except Exception as e:
        logging.error(f"Generation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# ---------- Helper Functions ----------
def extract_json_from_text(text: str) -> Optional[Dict]:
    # Try markdown code blocks first
    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, flags=re.S)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except Exception:
            pass
    
    # Fallback: raw JSON
    m = re.search(r'\{.*\}', text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        cleaned = re.sub(r"(['\"])?:\s*'([^']*)'", r'"\1": "\2"', m.group(0))
        try:
            return json.loads(cleaned)
        except Exception:
            return None

def truncate_prompt_for_sdxl(prompt: str, max_tokens: int = 75) -> str:
    if sd_tokenizer is None:
        words = prompt.split()
        return " ".join(words[:max_tokens]) if len(words) > max_tokens else prompt
    
    tokens = sd_tokenizer.encode(prompt)
    if len(tokens) <= max_tokens:
        return prompt
    
    truncated_tokens = tokens[:max_tokens]
    return sd_tokenizer.decode(truncated_tokens, skip_special_tokens=True)

def save_generated_image(image: Image.Image) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mental_health_{timestamp}_{uuid.uuid4().hex[:8]}.png"
    filepath = IMAGE_DIR / filename
    image.save(filepath)
    return f"/outputs/images/{filename}"

def save_uploaded_file(upload_file: UploadFile) -> str:
    """Save uploaded image and return path"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"upload_{timestamp}_{uuid.uuid4().hex[:8]}_{upload_file.filename}"
    filepath = UPLOAD_DIR / filename
    
    with open(filepath, "wb") as f:
        f.write(upload_file.file.read())
    
    return str(filepath)

# ---------- Prompts ----------
INTENT_PROMPT = """
系统: 你是高性能的意图解析与槽位抽取器（中文）。
用户: 我给你一些碎片化的用户输入{image_context}，请解析出明确写作意图、目标输出类型、以及关键槽位（audience, length, tone, require_references）。
输入碎片:
\"\"\"{fragments}\"\"\"

请仅返回合法 JSON，格式如下:
{{
  "intent": "...",
  "output_type": "...",
  "slots": {{
     "audience": "...",
     "length": "...",
     "tone": "...",
     "require_references": true/false
  }}
}}
"""

OUTLINE_PROMPT = """
系统: 你是心理科普大纲生成器（中文）。
用户: 依据意图与槽位{image_context}，生成 3-6 步可执行的大纲，每一步一句话并附 10-30 字说明。返回 JSON: {{ "outline": ["step1|说明","step2|说明", ...] }}
意图: {intent}
槽位: {slots}
"""

FEWSHOT_STYLE_PROMPT = """
系统: 你是风格化段落生成器（中文）。
说明: 给定示例风格和一段大纲{image_context}，请把大纲转换成目标风格的段落（200-400字），保持科学性且避免诊断性措辞。
示例风格:
{style_examples}

目标风格: {target_style}
大纲段: {outline_item}

请生成完整的段落内容，确保内容充实、连贯且有价值。只返回纯文本段落，不要包含额外说明或元数据。
"""

CRITIC_PROMPT = """
系统: 你是审稿者（Critic Agent），专注于心理健康文本的安全性与正确性（中文）。
任务: 检测文本中的下列风险：
 - 误导性医学或诊断性表述
 - 鼓励自伤/自杀或危险行为
 - 侮辱性、污名化或不当标签化语言
 - 泄露个人隐私或可识别信息（PII）
返回 JSON:
{{"flagged": bool, "issues": ["..."], "suggestions": ["..."], "safe_text": "..."}}

待检文本:
\"\"\"{text}\"\"\"
"""

IMAGE_PROMPT_TEMPLATE = """
Create a short English prompt (max 50 words) for Stable Diffusion XL.
Theme: {theme}
Style: {style}
Composition: {composition}

Return JSON: {{"image_prompt": "...", "note_cn": "..."}}
"""

# ---------- Safety Middleware ----------
async def safety_dependency(req: Request):
    body = await req.json()
    check_fields = []
    if 'fragments' in body: check_fields.append(body['fragments'])
    if 'outline' in body:
        if isinstance(body['outline'], list):
            check_fields.extend(body['outline'])
        else:
            check_fields.append(str(body['outline']))
    if 'text' in body: check_fields.append(body['text'])
    aggregated = "\n".join([str(c) for c in check_fields if c])
    issues = safety_utils.heuristic_check_text(aggregated)
    if issues:
        raise HTTPException(status_code=422, detail={"safety_issues": issues})

# ---------- Endpoints ----------
@app.post("/intent", dependencies=[Depends(safety_dependency)])
def extract_intent(req: IntentRequest):
    image_context = " 和一张图片" if req.image_path else ""
    prompt = INTENT_PROMPT.format(
        fragments=req.fragments,
        image_context=image_context
    )
    raw = generate_with_qwen3vl(prompt, image_path=req.image_path, max_tokens=256, temperature=0.0)
    parsed = extract_json_from_text(raw)
    if parsed is None:
        return {
            "intent": req.fragments[:60],
            "output_type": "short_post",
            "slots": {
                "audience": "general",
                "length": "short",
                "tone": "neutral",
                "require_references": False
            }
        }
    return parsed

@app.post("/outline", dependencies=[Depends(safety_dependency)])
def make_outline(req: OutlineRequest):
    image_context = "（参考提供的图片内容）" if req.image_path else ""
    prompt = OUTLINE_PROMPT.format(
        intent=req.intent,
        slots=json.dumps(req.slots, ensure_ascii=False),
        image_context=image_context
    )
    raw = generate_with_qwen3vl(prompt, image_path=req.image_path, max_tokens=256, temperature=0.0)
    parsed = extract_json_from_text(raw)
    if parsed and "outline" in parsed:
        return parsed
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    return {"outline": lines[:6]}

@app.post("/generate", dependencies=[Depends(safety_dependency)])
def generate(req: GenerateRequest):
    paragraphs = []
    se = req.style_examples or [
        {"style_name": "温柔引导", "example": "当你感到焦虑时，先关注呼吸，做三次深呼吸：吸气-屏气-呼气。"},
        {"style_name": "学术科普", "example": "研究表明，规律睡眠可提升情绪调节能力，相关文献如..."}
    ]
    
    image_context = "（结合图片内容）" if req.image_path else ""
    
    for item in req.outline:
        prompt = FEWSHOT_STYLE_PROMPT.format(
            style_examples=json.dumps(se, ensure_ascii=False),
            target_style=req.tone,
            outline_item=item,
            image_context=image_context
        )
        out = generate_with_qwen3vl(prompt, image_path=req.image_path, max_tokens=800, temperature=0.7)
        paragraphs.append(out.strip())

    full_text = "\n\n".join(paragraphs)

    # Critic check
    critic_out = generate_with_qwen3vl(
        CRITIC_PROMPT.format(text=full_text),
        max_tokens=512,
        temperature=0.0
    )
    parsed_critic = extract_json_from_text(critic_out) or {}
    flagged = parsed_critic.get("flagged", False)
    
    if flagged:
        safe_text = parsed_critic.get("safe_text", "")
        return {
            "flagged": True,
            "issues": parsed_critic.get("issues", []),
            "suggestions": parsed_critic.get("suggestions", []),
            "text": safe_text or full_text
        }
    return {"flagged": False, "text": full_text}

@app.post("/critic", dependencies=[Depends(safety_dependency)])
def critic(req: CriticRequest):
    out = generate_with_qwen3vl(
        CRITIC_PROMPT.format(text=req.text),
        max_tokens=512,
        temperature=0.0
    )
    return extract_json_from_text(out) or {}

@app.post("/image", dependencies=[Depends(safety_dependency)])
def make_image(req: ImageRequest):
    prompt = IMAGE_PROMPT_TEMPLATE.format(
        theme=req.theme,
        style=req.style,
        composition=req.composition
    )
    
    logging.info(f"=== IMAGE PROMPT GENERATION ===")
    logging.info(f"Theme: {req.theme}")
    
    image_prompt_json = generate_with_qwen3vl(
        prompt,
        max_tokens=150,
        temperature=0.2
    )
    
    logging.info(f"Raw output: {image_prompt_json}")
    
    parsed = extract_json_from_text(image_prompt_json)
    
    if parsed and "image_prompt" in parsed:
        img_prompt = parsed["image_prompt"]
    else:
        img_prompt = f"warm illustration, calm person, cozy room, soft colors, minimal background"
    
    # Truncate for SDXL
    img_prompt_truncated = truncate_prompt_for_sdxl(img_prompt, max_tokens=75)
    
    if sd_pipe is None:
        return {
            "image_url": None,
            "image_prompt": img_prompt_truncated,
            "note_cn": parsed.get("note_cn", "") if parsed else "",
            "warning": "SD model not loaded"
        }
    
    try:
        image = sd_pipe(
            img_prompt_truncated,
            num_inference_steps=25,
            guidance_scale=7.5,
            height=768,
            width=1024
        ).images[0]
        
        image_url = save_generated_image(image)
        
        logging.info(f"Image generated: {image_url}")
        
        return {
            "image_url": image_url,
            "image_prompt": img_prompt_truncated,
            "note_cn": parsed.get("note_cn", "") if parsed else "",
            "original_prompt": img_prompt if img_prompt != img_prompt_truncated else None
        }
    except Exception as e:
        logging.error(f"Image generation error: {e}")
        return {
            "image_url": None,
            "image_prompt": img_prompt_truncated,
            "note_cn": parsed.get("note_cn", "") if parsed else "",
            "error": str(e)
        }

# ---------- Upload Image Endpoint ----------
@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image to be used as context for content generation
    Returns the path to the uploaded image
    """
    try:
        image_path = save_uploaded_file(file)
        logging.info(f"Image uploaded: {image_path}")
        
        return {
            "image_path": image_path,
            "filename": file.filename
        }
        
    except Exception as e:
        logging.error(f"Image upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# ---------- Health Check ----------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "qwen_vl": QWEN_VL,
        "sd_model": SD_MODEL,
        "image_dir": str(IMAGE_DIR),
        "upload_dir": str(UPLOAD_DIR),
        "models_loaded": {
            "qwen3_vl": vl_model is not None,
            "processor": processor is not None,
            "sd_pipe": sd_pipe is not None
        },
        "capabilities": {
            "text_generation": True,
            "image_context": True,
            "image_generation": sd_pipe is not None,
            "multimodal": True
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8004)), log_level="info")