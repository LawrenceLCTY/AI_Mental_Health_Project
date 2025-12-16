# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import logging

# Transformers imports
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSequenceClassification

# Diffusers imports for image generation
from diffusers import StableDiffusionPipeline
import torch

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="AIMH: Multi-style Mental Health Content Agent")

# ========== Config: model names (change to local/quantized as needed) ==========
TEXT_MODEL = os.getenv("TEXT_MODEL", "Genius-Society/chatglm_6b")   # ChatGLM-6B for Chinese generation
STYLE_MODEL = TEXT_MODEL  # reuse ChatGLM for few-shot styleing
CRITIC_MODEL = os.getenv("CRITIC_MODEL", "google/bert-base-chinese") # used for tokenization/aux tasks
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "stable-diffusion-v1-5") # Stable Diffusion v1.5 on HF
# ===========================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# Load tokenizer + model (Causal LM for generation)
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, trust_remote_code=True)
text_model = AutoModelForCausalLM.from_pretrained(TEXT_MODEL, trust_remote_code=True, low_cpu_mem_usage=True)
text_model.to(device)
text_pipe = pipeline("text-generation", model=text_model, tokenizer=tokenizer, device=0 if device=="cuda" else -1)

# Critic / sequence classification (optional helper)
# We keep this simple: use a BERT tokenizer for auxiliary checks
critic_tokenizer = AutoTokenizer.from_pretrained(CRITIC_MODEL)
# Optionally load a classifier if you have one fine-tuned for safety; otherwise we use heuristics
#classifier = AutoModelForSequenceClassification.from_pretrained("some-safety-classifier") 

# Stable Diffusion pipeline (text->image)
# NOTE: you must have access token set in HF_HOME or HUGGINGFACE_TOKEN env var if model requires it.
pipe = None
try:
    pipe = StableDiffusionPipeline.from_pretrained(IMAGE_MODEL, torch_dtype=torch.float16).to(device)
except Exception as e:
    logging.warning("Stable Diffusion could not be loaded — image endpoints will be disabled. Error: %s", e)
    pipe = None

# ========== Pydantic request/response models ==========
class IntentRequest(BaseModel):
    fragments: str  # messy input fragments / user notes in Chinese
    user_profile: Optional[Dict[str,Any]] = None

class OutlineResponse(BaseModel):
    outline: List[str]
    intent_slots: Dict[str,str]

class GenerateRequest(BaseModel):
    outline: List[str]
    style_examples: Optional[List[Dict[str,str]]] = None
    tone: Optional[str] = "温和且专业"
    length: Optional[str] = "短文"

class GenerateResponse(BaseModel):
    text: str
    images: Optional[List[str]] = None  # data-urls or filepaths

class CriticRequest(BaseModel):
    text: str
    constraints: Optional[List[str]] = ["avoid medical diagnosis","no self-harm instructions","sensitive topics highlight"]

class CriticResponse(BaseModel):
    flagged: bool
    issues: List[str]
    corrected: Optional[str] = None

# ========== Helper prompt templates ==========
INTENT_PROMPT = """你是一个意图识别助手。用户输入（碎片化）内容如下：
{fragments}

请从中抽取：
1) 明确的写作意图（用一句话）
2) 需要的输出类型（如：短推文、科普长文、大纲、图文并茂推文）
3) 关键槽位（例如目标受众、目标长度、情绪基调、是否需要引用研究/数据）
返回JSON：{{"intent":"...", "output_type":"...", "slots":{{...}}}}
"""

OUTLINE_PROMPT = """你是心理科普写作助理。意图：{intent}。槽位：{slots}
基于这些碎片，生成一个按序列化的写作大纲（每步一句话，3-8步），每个步骤带上简短说明（10-30字）。
示例风格：清晰、易懂、对非专业读者友好。
输出JSON：{{"outline":["step1: ...","step2: ...", ...]}}
"""

FEWSHOT_STYLE_PROMPT = """你将把大纲段落扩展为目标风格的段落。
风格示例（少样本），每项包含 "style_name" 和 "example"：
{style_examples}

目标风格: {target_style}
大纲段: {outline_item}
生成要求：保持心理健康科学性、避免诊断性语言、给出实用建议或资源（如适当），语句不超过200字。
返回纯文本段落。
"""

CRITIC_PROMPT = """你是一个审稿者（Critic Agent），任务是检测文本中可能的错误与风险：
- 是否含有误导性医学建议或诊断
- 是否鼓励自伤/自杀/危险行为
- 是否有伦理问题或不当表述（耻辱化、标签化）
请列出找到的问题（若无则返回空列表），并给出修改建议（简短）。最后提供一个安全化的替代表述（如果需要）。
输入文本：
{text}
输出JSON：{{"flagged": bool, "issues":["..."], "suggestions":["..."], "safe_text":"..."}}
"""

IMAGE_PROMPT_TEMPLATE = """生成用于心理科普社交媒体的配图：
主题：{theme}
风格：{style}（如：简洁插画、温暖色调、扁平化、卡通人物）
构图要点：{composition}
禁止内容：真实创伤照片、人脸细节近照、医疗误导文本
生成英文/中文混合的提示词，最终用于 Stable Diffusion。
"""

# ========== Endpoints ==========

@app.post("/intent", response_model=OutlineResponse)
def extract_intent(req: IntentRequest):
    prompt = INTENT_PROMPT.format(fragments=req.fragments)
    # use text pipeline to parse JSON output
    out = text_pipe(prompt, max_length=512, do_sample=False)[0]['generated_text']
    # Attempt to parse JSON from model output (best-effort)
    import json, re
    match = re.search(r'\{.*\}', out, re.S)
    if not match:
        # fallback: minimal heuristic: put fragments back as single-step outline
        return OutlineResponse(outline=[req.fragments], intent_slots={"intent":"unknown"})
    try:
        parsed = json.loads(match.group(0))
        intent = parsed.get("intent","")
        output_type = parsed.get("output_type","short_post")
        slots = parsed.get("slots",{})
        # create a minimal outline hint
        outline = [f"根据意图生成：{intent}", "提供背景/定义", "给出3条实用建议", "结尾与资源推荐"]
        return OutlineResponse(outline=outline, intent_slots={"intent":intent, "output_type":output_type, **slots})
    except Exception as e:
        return OutlineResponse(outline=[req.fragments], intent_slots={"intent":"parse_error"})

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    # Step 1: Expand each outline item into paragraphs according to style examples
    paragraphs = []
    for item in req.outline:
        style_examples = req.style_examples or []
        prompt = FEWSHOT_STYLE_PROMPT.format(
            style_examples=str(style_examples),
            target_style=req.tone,
            outline_item=item
        )
        out = text_pipe(prompt, max_length=512, do_sample=True, temperature=0.7)[0]['generated_text']
        # strip prompt
        paragraphs.append(out.strip())
    full_text = "\n\n".join(paragraphs)

    # Step 2: Critic check (safety/correctness)
    crit = critic_check_internal(full_text)
    if crit.flagged:
        # If flagged, prefer returning corrected safe_text
        safe_text = crit.corrected or full_text
        response_text = f"[CRITIC NOTES]\n{'; '.join(crit.issues)}\n\n[SAFE VERSION]\n{safe_text}"
    else:
        response_text = full_text

    # Step 3: Optionally generate images (one image per theme)
    images = []
    if pipe is not None:
        # Use the first outline item as theme for the image prompt (simplified)
        theme = req.outline[0] if len(req.outline)>0 else "心理健康"
        image_prompt = IMAGE_PROMPT_TEMPLATE.format(theme=theme, style="简洁插画, 温暖色调", composition="single friendly character, calm setting")
        # Prefer English short prompt for SD
        sd_prompt = "A warm, friendly flat-style illustration of a calm person holding a cup, soft warm color palette, clean background, social media banner"
        image = pipe(sd_prompt, guidance_scale=7.5, num_inference_steps=20).images[0]
        # Save image to disk and return filepath
        out_path = f"/tmp/aimh_image_{abs(hash(sd_prompt))%10_000}.png"
        image.save(out_path)
        images.append(out_path)

    return GenerateResponse(text=response_text, images=images)

@app.post("/critic", response_model=CriticResponse)
def critic_check(req: CriticRequest):
    cr = critic_check_internal(req.text, req.constraints)
    return cr

# ========== Internal critic implementation (heuristic + LM) ==========
def critic_check_internal(text: str, constraints: Optional[List[str]] = None) -> CriticResponse:
    # 1) quick heuristic checks
    issues = []
    flagged = False
    if "自杀" in text or "自杀" in text:
        issues.append("文本中出现自杀/自伤术语 — 需要安全处理（触发危机干预流程）")
        flagged = True
    # 2) run critic LM prompt to get structured response
    critic_prompt = CRITIC_PROMPT.format(text=text)
    try:
        out = text_pipe(critic_prompt, max_length=512, do_sample=False)[0]['generated_text']
    except Exception:
        out = ""
    # parse JSON if present
    import re, json
    match = re.search(r'\{.*\}', out, re.S)
    corrected = None
    if match:
        try:
            parsed = json.loads(match.group(0))
            issues += parsed.get("issues",[])
            if parsed.get("flagged", False):
                flagged = True
            corrected = parsed.get("safe_text")
        except Exception:
            pass
    return CriticResponse(flagged=flagged, issues=list(set(issues)), corrected=corrected)

# ========== Run ==========
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)
