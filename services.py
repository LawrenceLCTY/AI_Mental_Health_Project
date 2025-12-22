# services.py
import os
import json
import time
from google import genai
from google.genai import types
from openai import OpenAI
from typing import Generator, Optional, Dict, Any
from models import ContentBlueprint, CreativeBrief
from prompts import (
    GEMINI_BRIEF_PROMPT,
    DEEPSEEK_BRIEF_PROMPT,
    OUTLINE_GENERATION_PROMPT,
    SECTION_TEXT_PROMPT
)
from templates import TEMPLATE_COGNITIVE_FLIP
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image as PILImage
from dotenv import load_dotenv
load_dotenv()

class LLMService:
    def __init__(self, provider: str, model_name: str, api_key: Optional[str] = None):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.client = None
        self.local_model = None
        self.local_tokenizer = None
        self.supports_images = False  # Track if provider supports image input
        self.sd_pipeline = None  # Stable Diffusion pipeline for image generation
        
        # Set PyTorch memory configuration for better memory management
        if not os.getenv("PYTORCH_CUDA_ALLOC_CONF"):
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        # Initialize Stable Diffusion if model path is available
        self.sd_model_path = None
        self._init_stable_diffusion()
        
        if provider == "DeepSeek":
            # if not api_key or api_key.strip() == "":
            #     raise ValueError("DeepSeek 需要提供 API Key")
            self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        elif provider == "Google Gemini":
            # 新版 SDK 初始化方式
            # 优先使用传入的 key，否则自动查找环境变量 GEMINI_API_KEY
            key_to_use = api_key if api_key else os.getenv("GEMINI_API_KEY")
            if key_to_use:
                self.client = genai.Client(api_key=key_to_use)
            else:
                # 尝试无参初始化（依赖环境变量）
                self.client = genai.Client()
        elif provider == "Qwen-VL":
            try:
                # Load Qwen3-VL model using the official approach
                print(f"Loading Qwen3-VL model from: {self.model_name}")
                
                # Load processor (replaces tokenizer for VL models)
                self.local_tokenizer = AutoProcessor.from_pretrained(self.model_name)
                
                # Load model with Qwen3VL-specific class
                self.local_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
                
                # Qwen3-VL supports image input
                self.supports_images = True
                
                print(f"✓ Qwen3-VL model loaded successfully")
                
            except Exception as e:
                raise Exception(f"加载本地模型失败: {str(e)}")
    
    def _init_stable_diffusion(self):
        """Initialize Stable Diffusion XL pipeline for image generation
        
        Note: Pipeline is loaded on-demand to save memory when using large VL models.
        """
        try:
            sd_model_path = os.getenv("SD_MODEL")
            
            if not sd_model_path:
                print("⚠️  SD_MODEL environment variable not set. Image generation disabled.")
                return
            
            if not os.path.exists(sd_model_path):
                print(f"⚠️  SD model path does not exist: {sd_model_path}")
                return
            
            # Store path but don't load yet - will load on-demand
            self.sd_model_path = sd_model_path
            self.sd_pipeline = None
            print("✓ Stable Diffusion path configured (will load on-demand)")
            
        except Exception as e:
            print(f"⚠️  Failed to configure Stable Diffusion: {str(e)}")
            self.sd_model_path = None
            self.sd_pipeline = None
    
    def _load_sd_pipeline(self):
        """Load Stable Diffusion pipeline on-demand"""
        if self.sd_pipeline is not None:
            return  # Already loaded
        
        if not hasattr(self, 'sd_model_path') or not self.sd_model_path:
            return
        
        try:
            print("Loading Stable Diffusion XL pipeline...")
            
            # If Qwen model is loaded on GPU, move SD to CPU to avoid OOM
            use_cpu = False
            if self.local_model is not None and torch.cuda.is_available():
                # Check available GPU memory
                gpu_memory_free = torch.cuda.mem_get_info()[0] / 1024**3  # GB
                if gpu_memory_free < 8:  # Need at least 8GB free for SDXL
                    print(f"⚠️  Low GPU memory ({gpu_memory_free:.2f}GB free). Loading SD on CPU.")
                    use_cpu = True
            
            self.sd_pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.sd_model_path,
                torch_dtype=torch.float16 if not use_cpu else torch.float32,
                use_safetensors=True,
                variant="fp16" if not use_cpu else None
            )
            
            if use_cpu or not torch.cuda.is_available():
                self.sd_pipeline = self.sd_pipeline.to("cpu")
                print("✓ Stable Diffusion XL loaded on CPU")
            else:
                self.sd_pipeline = self.sd_pipeline.to("cuda")
                print("✓ Stable Diffusion XL loaded on GPU")
            
            # Enable memory efficient attention
            self.sd_pipeline.enable_attention_slicing()
            
        except Exception as e:
            print(f"⚠️  Failed to load Stable Diffusion: {str(e)}")
            self.sd_pipeline = None

    def _translate_brief_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Translate Chinese field names to English for CreativeBrief model
        
        This handles cases where Qwen-VL generates Chinese field names despite instructions.
        """
        
        # Comprehensive mapping of Chinese to English field names
        field_mapping = {
            # Top-level fields
            '定位靶心': 'targeting',
            '核心洞察': 'insight',
            '价值跨越': 'transformation',
            '沟通策略': 'strategy',
            
            # Transformation fields
            '当前状态': 'current_state',
            '现状': 'current_state',
            '期望状态': 'desired_state',
            '愿景': 'desired_state',
            
            # Strategy fields
            '钩子类型': 'hook_type',
            '钩子': 'hook_type',
            '沟通语气': 'tone',
            '语气': 'tone',
            '人设': 'tone',
            
            # Already English (pass through)
            'targeting': 'targeting',
            'insight': 'insight',
            'transformation': 'transformation',
            'current_state': 'current_state',
            'desired_state': 'desired_state',
            'strategy': 'strategy',
            'hook_type': 'hook_type',
            'tone': 'tone'
        }
        
        translated = {}
        
        for key, value in data.items():
            # Translate key if it's in mapping, otherwise keep original
            english_key = field_mapping.get(key, key)
            
            # If value is a dict, recursively translate
            if isinstance(value, dict):
                translated[english_key] = self._translate_brief_fields(value)
            else:
                translated[english_key] = value
        
        return translated

    def _generate_with_qwen(self, prompt: str, image_path: Optional[str] = None, max_tokens: int = 1024, temperature: float = 0.9) -> str:
        """Helper method to generate text with Qwen3-VL model (supports text-only or text+image)"""
        try:
            # Prepare content list
            content = []
            
            # Add image if provided
            if image_path:
                from PIL import Image
                # Load image
                image = Image.open(image_path)
                content.append({"type": "image", "image": image})
            
            # Add text prompt
            content.append({"type": "text", "text": prompt})
            
            # Format messages for Qwen3-VL
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
            
            # Apply chat template with tokenization
            # When image is provided, processor handles it properly
            inputs = self.local_tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.local_model.device)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.local_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                )
            
            # Trim input tokens from output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]
            
            # Decode using processor
            output_text = self.local_tokenizer.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            return output_text[0].strip() if output_text else ""
            
        except Exception as e:
            raise Exception(f"Qwen-VL generation failed: {str(e)}")

    def generate_creative_brief(self, fragments: str, image_path: Optional[str] = None) -> Optional[CreativeBrief]:
        """第一步：生成创意简报"""
        if not (self.client or (self.local_model and self.local_tokenizer)):
            raise Exception(f"{self.provider} 客户端未正确初始化")
            
        try:
            if self.provider == "Google Gemini":
                prompt = GEMINI_BRIEF_PROMPT.format(fragments=fragments)
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=CreativeBrief
                    )
                )
                return CreativeBrief(**json.loads(response.text))

            elif self.provider == "DeepSeek":
                prompt = DEEPSEEK_BRIEF_PROMPT.format(fragments=fragments)
                schema_hint = CreativeBrief.model_json_schema()
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": f"{prompt}\nJSON Schema: {json.dumps(schema_hint)}"}
                    ],
                    response_format={'type': 'json_object'},
                    temperature=1.3
                )
                
                content = response.choices[0].message.content
                if not content:
                    raise Exception("DeepSeek 返回空内容")
                
                # 清理可能的 markdown 代码块标记
                content = content.strip()
                if content.startswith('```json'):
                    content = content[7:]
                if content.startswith('```'):
                    content = content[3:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()
                
                # 解析并验证数据
                data = json.loads(content)
                return CreativeBrief(**data)
            
            elif self.provider == "Qwen-VL":
                # Use the GEMINI brief prompt as the base for local generation
                base_prompt = GEMINI_BRIEF_PROMPT.format(fragments=fragments)
                
                # Add explicit JSON schema instruction for Qwen-VL with bilingual clarity
                schema_instruction = """
===== 【输出格式要求】CRITICAL OUTPUT FORMAT =====
你必须输出JSON格式，但是字段名（keys）必须用英文，内容（values）用中文。

Required JSON structure (字段名用英文，内容用中文):
{
  "targeting": "你的定位靶心内容（用中文写内容，但字段名必须是英文targeting）",
  "insight": "你的核心洞察内容（用中文写内容，但字段名必须是英文insight）",
  "transformation": {
    "current_state": "当前状态内容（用中文写内容，但字段名必须是英文current_state）",
    "desired_state": "期望状态内容（用中文写内容，但字段名必须是英文desired_state）"
  },
  "strategy": {
    "hook_type": "钩子类型内容（用中文写内容，但字段名必须是英文hook_type）",
    "tone": "沟通语气内容（用中文写内容，但字段名必须是英文tone）"
  }
}

❌ 错误示例（不要这样）:
{
  "定位靶心": "...",  // 字段名不能用中文！
  "核心洞察": "..."
}

✅ 正确示例:
{
  "targeting": "因为写不出论文而焦虑失眠的博士生",  // 字段名用英文，内容用中文
  "insight": "降低心理预期，先写垃圾初稿",
  "transformation": {
    "current_state": "陷入完美主义陷阱，一个字都写不出来",
    "desired_state": "接受初稿可以很烂，开始动笔"
  },
  "strategy": {
    "hook_type": "反直觉：完美主义是拖延的罪魁祸首",
    "tone": "理性学霸，一针见血"
  }
}

重要提醒：
- 字段名称（keys）: targeting, insight, transformation, current_state, desired_state, strategy, hook_type, tone
- 不要使用: "定位靶心", "核心洞察", "价值跨越", "当前状态", "期望状态", "沟通策略", "钩子类型", "沟通语气"
===== END OF FORMAT =====
"""
                
                prompt = base_prompt + "\n\n" + schema_instruction
                
                # Generate with the local model (pass image_path if provided)
                text = self._generate_with_qwen(prompt, image_path=image_path, max_tokens=1024, temperature=0.9)
                
                content = text.strip()
                # strip possible code fences
                if content.startswith('```json'):
                    content = content[7:]
                if content.startswith('```'):
                    content = content[3:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()
                
                data = json.loads(content)
                
                # Debug: Log original structure before translation
                print(f"[DEBUG] Qwen-VL raw output keys: {list(data.keys())}")
                if 'transformation' in data:
                    print(f"[DEBUG] transformation keys: {list(data['transformation'].keys())}")
                elif '价值跨越' in data:
                    print(f"[DEBUG] Found Chinese '价值跨越', keys: {list(data['价值跨越'].keys())}")
                if 'strategy' in data:
                    print(f"[DEBUG] strategy keys: {list(data['strategy'].keys())}")
                elif '沟通策略' in data:
                    print(f"[DEBUG] Found Chinese '沟通策略', keys: {list(data['沟通策略'].keys())}")
                
                # Translate Chinese field names to English if model still used them
                data = self._translate_brief_fields(data)
                
                print(f"[DEBUG] After translation keys: {list(data.keys())}")
                
                return CreativeBrief(**data)
            else:
                raise Exception(f"未知的服务商: {self.provider}")

        except Exception as e:
            raise Exception(f"创意简报生成失败: {str(e)}")

    def generate_blueprint(self, fragments: str, style: str, image_path: Optional[str] = None) -> Optional[ContentBlueprint]:
        """第二步：基于创意简报和模板生成大纲"""
        
        # 先生成创意简报
        brief = self.generate_creative_brief(fragments, image_path=image_path)
        if not brief:
            return None
        
        # 准备数据：将创意简报和模板序列化为 JSON
        brief_json = json.dumps({
            "targeting": brief.targeting,
            "insight": brief.insight,
            "current_state": brief.transformation.current_state,
            "desired_state": brief.transformation.desired_state,
            "hook_type": brief.strategy.hook_type,
            "tone": brief.strategy.tone
        }, ensure_ascii=False, indent=2)
        
        template_json = json.dumps(TEMPLATE_COGNITIVE_FLIP, ensure_ascii=False, indent=2)
        
        # 使用新的大纲生成 prompt
        prompt = OUTLINE_GENERATION_PROMPT.format(
            brief_json=brief_json,
            template_json=template_json
        )

        try:
            if self.provider == "Google Gemini":
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                outline_data = json.loads(response.text)
                
                # 转换模板输出为 OutlineSection 格式
                sections = []
                if 'outline' in outline_data:
                    for idx, section in enumerate(outline_data['outline'], 1):
                        # 从模板查找对应的元数据
                        template_section = next(
                            (s for s in TEMPLATE_COGNITIVE_FLIP['structure'] if s['section_id'] == section.get('section_id')),
                            None
                        )
                        
                        sections.append({
                            'id': idx,
                            'title': section.get('title', ''),
                            'intent': template_section['content_instruction'] if template_section else section.get('draft_content', ''),
                            'key_points': [section.get('draft_content', '')]  # 将draft_content作为关键点
                        })
                
                # Pass dict data directly, let Pydantic instantiate OutlineSection
                return ContentBlueprint(brief=brief, outline=sections)

            elif self.provider == "DeepSeek":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    response_format={'type': 'json_object'},
                    temperature=1.3
                )
                outline_data = json.loads(response.choices[0].message.content)
                
                # 转换模板输出为 OutlineSection 格式
                sections = []
                if 'outline' in outline_data:
                    for idx, section in enumerate(outline_data['outline'], 1):
                        # 从模板查找对应的元数据
                        template_section = next(
                            (s for s in TEMPLATE_COGNITIVE_FLIP['structure'] if s['section_id'] == section.get('section_id')),
                            None
                        )
                        
                        sections.append({
                            'id': idx,
                            'title': section.get('title', ''),
                            'intent': template_section['content_instruction'] if template_section else section.get('draft_content', ''),
                            'key_points': [section.get('draft_content', '')]  # 将draft_content作为关键点
                        })
                
                # Pass dict data directly, let Pydantic instantiate OutlineSection
                return ContentBlueprint(brief=brief, outline=sections)

            elif self.provider == "Qwen-VL":
                # Add explicit JSON schema instruction for Qwen-VL
                schema_instruction = """
You MUST respond with a JSON object using EXACTLY these English field names:
{
  "outline": [
    {
      "section_id": "string",
      "title": "string", 
      "draft_content": "string"
    }
  ]
}

IMPORTANT: Use ONLY English field names. Do not translate to Chinese.
"""
                
                prompt_with_schema = prompt + "\n\n" + schema_instruction
                
                # Local generation for outline JSON
                text = self._generate_with_qwen(prompt_with_schema, max_tokens=1500, temperature=0.9)
                
                content = text.strip()
                # try to clean fences
                if content.startswith('```json'):
                    content = content[7:]
                if content.startswith('```'):
                    content = content[3:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()
                
                outline_data = json.loads(content)

                # 转换模板输出为 OutlineSection 格式
                sections = []
                if 'outline' in outline_data:
                    for idx, section in enumerate(outline_data['outline'], 1):
                        # 从模板查找对应的元数据
                        template_section = next(
                            (s for s in TEMPLATE_COGNITIVE_FLIP['structure'] if s['section_id'] == section.get('section_id')),
                            None
                        )

                        sections.append({
                            'id': idx,
                            'title': section.get('title', ''),
                            'intent': template_section['content_instruction'] if template_section else section.get('draft_content', ''),
                            'key_points': [section.get('draft_content', '')]
                        })

                # Pass dict data directly, let Pydantic instantiate OutlineSection
                return ContentBlueprint(brief=brief, outline=sections)

        except Exception as e:
            raise Exception(f"大纲生成失败: {str(e)}")

    def generate_section_text(self, section: 'OutlineSection', brief: CreativeBrief, section_idx: int) -> str:
        """第三步：基于模板和大纲生成单章节正文"""
        
        # 从模板中获取对应章节的元数据
        template_sections = TEMPLATE_COGNITIVE_FLIP['structure']
        template_section = template_sections[section_idx] if section_idx < len(template_sections) else None
        
        if not template_section:
            return "[错误: 无法找到对应的模板章节]"
        
        # 准备 prompt 参数
        prompt = SECTION_TEXT_PROMPT.format(
            section_role=template_section['role'],
            section_title=section.title,
            content_instruction=template_section['content_instruction'],
            word_count_limit=template_section.get('word_count_limit', 200),
            targeting=brief.targeting,
            insight=brief.insight,
            current_state=brief.transformation.current_state,
            desired_state=brief.transformation.desired_state,
            tone=brief.strategy.tone
        )

        try:
            if self.provider == "Google Gemini":
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="text/plain"
                    )
                )
                return response.text if response.text else ""

            elif self.provider == "DeepSeek":
                # === 修复：DeepSeek 关闭流式 ===
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False  # 显式关闭流式
                )
                return response.choices[0].message.content

            elif self.provider == "Qwen-VL":
                # Calculate appropriate max tokens based on word count limit
                max_tokens = template_section.get('word_count_limit', 200) * 3
                
                text = self._generate_with_qwen(prompt, max_tokens=max_tokens, temperature=0.9)
                return text

        except Exception as e:
            return f"[生成出错: {str(e)}]"
    
    def generate_illustration(self, section: 'OutlineSection', brief: CreativeBrief, section_text: str) -> Optional[str]:
        """Generate an illustration image for a section using Stable Diffusion
        
        Args:
            section: The outline section
            brief: The creative brief for context
            section_text: The generated text content for this section
            
        Returns:
            Path to saved image file, or None if generation failed
        """
        # Load SD pipeline on-demand
        self._load_sd_pipeline()
        
        if not self.sd_pipeline:
            print("⚠️  Stable Diffusion not available")
            return None
        
        try:
            # Step 1: Use LLM to generate an image prompt based on the content
            image_prompt = self._generate_image_prompt(section, brief, section_text)
            
            if not image_prompt:
                return None
            
            print(f"🎨 Image prompt: {image_prompt}")
            
            # Step 2: Free up GPU memory if using Qwen-VL on GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Step 3: Generate image with Stable Diffusion
            # Add quality enhancing keywords
            enhanced_prompt = f"{image_prompt}, high quality, detailed, professional illustration, clean design"
            negative_prompt = "text, watermark, signature, blurry, low quality, distorted, ugly, bad anatomy"
            
            image = self.sd_pipeline(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                width=1024,
                height=1024
            ).images[0]
            
            
            # Step 5: Save image
            import tempfile
            output_dir = "outputs/images"
            os.makedirs(output_dir, exist_ok=True)
            
            image_filename = f"section_{section.id}_{int(time.time())}.png"
            image_path = os.path.join(output_dir, image_filename)
            image.save(image_path)
            
            print(f"✅ Image saved to: {image_path}")
            return image_path
            
        except Exception as e:
            print(f"⚠️  Image generation failed: {str(e)}")
            return None
    
    def _generate_image_prompt(self, section: 'OutlineSection', brief: CreativeBrief, section_text: str) -> Optional[str]:
        """Use LLM to generate a Stable Diffusion prompt based on the content"""
        
        prompt_generation_instruction = f"""
Based on the following content, generate a concise Stable Diffusion prompt for an illustration image.

Section Title: {section.title}
Section Content: {section_text[:500]}...
Core Insight: {brief.insight}
Tone: {brief.strategy.tone}

Requirements:
1. Describe a single, clear visual concept (not multiple scenes)
2. Use concrete visual elements (colors, objects, atmosphere)
3. Match the tone: {brief.strategy.tone}
4. Keep it under 50 words
5. Focus on symbolic or metaphorical representation
6. NO text, NO people's faces (use silhouettes if needed)

Output ONLY the image prompt, nothing else. Use English for better SD results.

Example good prompts:
- "A minimalist geometric staircase ascending into clouds, soft gradient sky, hope and progress concept"
- "Tangled red threads slowly untangling into organized lines, overhead view, problem-solving metaphor"
- "A single bright lightbulb illuminating a dark cluttered desk, contrast between chaos and clarity"
"""
        
        try:
            if self.provider == "Google Gemini":
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt_generation_instruction,
                    config=types.GenerateContentConfig(
                        response_mime_type="text/plain"
                    )
                )
                return response.text.strip() if response.text else None
            
            elif self.provider == "DeepSeek":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt_generation_instruction}],
                    stream=False,
                    temperature=0.8
                )
                return response.choices[0].message.content.strip()
            
            elif self.provider == "Qwen-VL":
                text = self._generate_with_qwen(prompt_generation_instruction, max_tokens=200, temperature=0.8)
                return text.strip()
            
            return None
            
        except Exception as e:
            print(f"⚠️  Image prompt generation failed: {str(e)}")
            return None