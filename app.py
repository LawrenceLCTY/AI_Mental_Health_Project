# app.py
import os
import re
import json
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
from PIL import Image
import torch

# transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# Qwen-VL may use a special class via trust_remote_code
from transformers import AutoProcessor
from diffusers import StableDiffusionPipeline  # fallback only if you prefer SD

# local helpers
import utils.safety as safety_utils
import utils.io_utils as io_utils

load_dotenv()
logging.basicConfig(level=logging.INFO)

# ---------- Config ----------
QWEN_INSTRUCT = os.getenv("QWEN_INSTRUCT", "qwen/qwen3-7b-instruct")   # example id
QWEN_VL = os.getenv("QWEN_VL", "qwen/qwen3-vl-7b")                     # example id

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Device: {DEVICE}")

# ---------- Load models ----------
# Text model (Qwen3-Instruct) - used for intent, outline, style, critic
tokenizer = AutoTokenizer.from_pretrained(QWEN_INSTRUCT, trust_remote_code=True, local_files_only=True)
text_model = AutoModelForCausalLM.from_pretrained(QWEN_INSTRUCT, trust_remote_code=True, torch_dtype=torch.float16 if DEVICE=="cuda" else None, local_files_only=True)
text_model.to(DEVICE)
text_pipe = pipeline("text-generation", model=text_model, tokenizer=tokenizer, device=0 if DEVICE=="cuda" else -1)

# Qwen3-VL: processor + model (for image generation tasks Qwen3-VL may expose text2im)
# We will try to load a multimodal pipeline. If not available, fallback to calling Qwen-VL's text generator
vl_processor = None
vl_pipeline = None
try:
    vl_processor = AutoProcessor.from_pretrained(QWEN_VL, trust_remote_code=True, local_files_only=True)
    # Many Qwen-VL distributions expose .from_pretrained to generate images via a .generate or special pipeline.
    # If a direct text->image is not available, we will generate image prompts with Qwen3-Instruct and call Stable Diffusion as fallback.
    # For simplicity, we use Qwen-VL for prompt crafting and SD for actual rendering if Qwen-VL rendering isn't available locally.
    # If your Qwen-VL supports .from_pretrained(..., device_map="auto") for text->image, replace below accordingly.
    logging.info("Loaded Qwen-VL processor (for multimodal prompts).")  
except Exception as e:
    logging.warning("Could not load Qwen-VL processor: %s", e)

# Optional: Stable Diffusion fallback for image rendering (if Qwen-VL not available for text->image)
SD_MODEL = os.getenv("SD_MODEL", "runwayml/stable-diffusion-v1-5")
sd_pipe = None
try:
    sd_pipe = StableDiffusionPipeline.from_pretrained(SD_MODEL, torch_dtype=torch.float16 if DEVICE=="cuda" else None, local_files_only=True).to(DEVICE)
    logging.info("Stable Diffusion fallback loaded.")
except Exception as e:
    logging.warning("Stable Diffusion not loaded (will skip image rendering): %s", e)

# ---------- FastAPI setup ----------
app = FastAPI(title="AIMH-Qwen3 Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class IntentRequest(BaseModel):
    fragments: str
    user_profile: Optional[Dict[str, Any]] = None

class OutlineRequest(BaseModel):
    intent: str
    slots: Dict[str, Any]

class GenerateRequest(BaseModel):
    outline: List[str]
    style_examples: Optional[List[Dict[str,str]]] = None
    tone: Optional[str] = "温和且专业"
    length: Optional[str] = "短文"

class CriticRequest(BaseModel):
    text: str
    require_safe: Optional[bool] = True

class ImageRequest(BaseModel):
    theme: str
    style: Optional[str] = "简洁插画, 温暖色调"
    composition: Optional[str] = "single character, calm setting, minimal background"

# ---------- Prompts (optimized for Qwen3) ----------
# Use instruction style: system:..., user:... is abstracted here since we pass plain text prompts.
INTENT_PROMPT = """
系统: 你是高性能的意图解析与槽位抽取器（中文）。
用户: 我给你一些碎片化的用户输入，请解析出明确写作意图、目标输出类型、以及关键槽位（audience, length, tone, require_references）。
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
用户: 依据意图与槽位，生成 3-6 步可执行的大纲，每一步一句话并附 10-30 字说明。返回 JSON: {{ "outline": ["step1|说明","step2|说明", ...] }}
意图: {intent}
槽位: {slots}
"""

FEWSHOT_STYLE_PROMPT = """
系统: 你是风格化段落生成器（中文，Qwen3-指令格式优化）。
说明: 给定示例风格和一段大纲，请把大纲转换成目标风格的段落（不超过200字），保持科学性且避免诊断性措辞。
示例风格:
{style_examples}

目标风格: {target_style}
大纲段: {outline_item}

只返回纯文本段落，不要包含额外说明或元数据。
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
系统: 你是图像提示工程师（中文）。
任务: 根据主题、风格与构图要求生成一个可直接用于图像生成器（Stable Diffusion 或 Qwen3-VL）的英文/中文混合提示词（短句），同时返回一段 1-2 句的备用说明（中文）供编辑。
主题: {theme}
风格: {style}
构图: {composition}
禁忌: 真实创伤照片、面部近照、医疗操作细节、任何血腥或恐怖元素

返回 JSON:
{{"image_prompt": "...", "note_cn": "..."}}
"""

# ---------- Utilities ----------
def run_text_generation(prompt: str, max_tokens: int = 512, temperature: float = 0.7, do_sample: bool = True) -> str:
    out = text_pipe(prompt, max_new_tokens=max_tokens, do_sample=do_sample, temperature=temperature)[0]["generated_text"]
    # If the model simply echoes the prompt, try to trim
    generated = out[len(prompt):] if out.startswith(prompt) else out
    return generated.strip()

def extract_json_from_text(text: str) -> Optional[Dict]:
    m = re.search(r'\{.*\}', text, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj
    except Exception:
        # try to sanitize trailing commas or single quotes
        cleaned = re.sub(r"(['\"])?:\s*'([^']*)'", r'"\1": "\2"', m.group(0))
        try:
            return json.loads(cleaned)
        except Exception:
            return None

# ---------- Safety middleware (dependency) ----------
async def safety_dependency(req: Request):
    body = await req.json()
    # quick heuristic checks on text fields in request
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
        # return structured rejection so front-end can surface to user and require confirmation
        raise HTTPException(status_code=422, detail={"safety_issues": issues})

# ---------- Endpoints (pipeline) ----------
@app.post("/intent", dependencies=[Depends(safety_dependency)])
def extract_intent(req: IntentRequest):
    prompt = INTENT_PROMPT.format(fragments=req.fragments)
    raw = run_text_generation(prompt, max_tokens=256, temperature=0.0, do_sample=False)
    parsed = extract_json_from_text(raw)
    if parsed is None:
        # fallback: minimal representation
        return {"intent": req.fragments[:60], "output_type": "short_post", "slots": {"audience":"general","length":"short","tone":"neutral","require_references":False}}
    return parsed

@app.post("/outline", dependencies=[Depends(safety_dependency)])
def make_outline(req: OutlineRequest):
    prompt = OUTLINE_PROMPT.format(intent=req.intent, slots=json.dumps(req.slots, ensure_ascii=False))
    raw = run_text_generation(prompt, max_tokens=256, temperature=0.0, do_sample=False)
    parsed = extract_json_from_text(raw)
    if parsed and "outline" in parsed:
        # split each outline item by '|' into title and note
        return parsed
    # fallback parse heuristics: split lines
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    return {"outline": lines[:6]}

@app.post("/generate", dependencies=[Depends(safety_dependency)])
def generate(req: GenerateRequest):
    paragraphs = []
    se = req.style_examples or [
        {"style_name":"温柔引导", "example":"当你感到焦虑时，先关注呼吸，做三次深呼吸：吸气-屏气-呼气。"},
        {"style_name":"学术科普", "example":"研究表明，规律睡眠可提升情绪调节能力，相关文献如..."}
    ]
    for item in req.outline:
        prompt = FEWSHOT_STYLE_PROMPT.format(
            style_examples=json.dumps(se, ensure_ascii=False),
            target_style=req.tone,
            outline_item=item
        )
        out = run_text_generation(prompt, max_tokens=300, temperature=0.7, do_sample=True)
        paragraphs.append(out.strip())

    full_text = "\n\n".join(paragraphs)

    # Critic check before returning
    critic_out = run_text_generation(CRITIC_PROMPT.format(text=full_text), max_tokens=256, temperature=0.0, do_sample=False)
    parsed_critic = extract_json_from_text(critic_out) or {}
    flagged = parsed_critic.get("flagged", False)
    if flagged:
        safe_text = parsed_critic.get("safe_text", "")
        return {"flagged": True, "issues": parsed_critic.get("issues", []), "suggestions": parsed_critic.get("suggestions", []), "text": safe_text or full_text}
    return {"flagged": False, "text": full_text}

@app.post("/critic", dependencies=[Depends(safety_dependency)])
def critic(req: CriticRequest):
    out = run_text_generation(CRITIC_PROMPT.format(text=req.text), max_tokens=256, temperature=0.0, do_sample=False)
    parsed = extract_json_from_text(out) or {}
    # ensure format
    return parsed

@app.post("/image", dependencies=[Depends(safety_dependency)])
def make_image(req: ImageRequest):
    # Step 1: craft image prompt with Qwen3-VL (or Qwen3-Instruct if VL unavailable)
    image_prompt_json = run_text_generation(IMAGE_PROMPT_TEMPLATE.format(theme=req.theme, style=req.style, composition=req.composition), max_tokens=200, temperature=0.2, do_sample=False)
    parsed = extract_json_from_text(image_prompt_json)
    if parsed and "image_prompt" in parsed:
        img_prompt = parsed["image_prompt"]
    else:
        # fallback short english stable-diffusion prompt
        img_prompt = f"A warm flat-style illustration of a calm person in a cozy room, soft warm colors, minimal background, social-media banner. Theme: {req.theme}"

    # Step 2: render image via SD fallback if available
    if sd_pipe is None:
        # If no renderer, return only prompt for external rendering (e.g. Qwen-VL online)
        return {"image_prompt": img_prompt, "note_cn": parsed.get("note_cn","") if parsed else ""}
    # generate
    image = sd_pipe(img_prompt, guidance_scale=7.5, num_inference_steps=20).images[0]
    out_path = io_utils.save_image_temp(image, prefix="aimh_qwen3")
    return {"image_path": out_path, "image_prompt": img_prompt, "note_cn": parsed.get("note_cn","") if parsed else ""}

# ---------- Health check ----------
@app.get("/health")
def health():
    return {"status":"ok", "device": DEVICE, "qwen_instruct": QWEN_INSTRUCT, "qwen_vl": QWEN_VL}

# ---------- Run via uvicorn ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8004)), log_level="info")
