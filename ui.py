import gradio as gr
import requests
import os
from pathlib import Path

API_URL = os.getenv("API_URL", "http://127.0.0.1:8004")
TIMEOUT = 999

def generate_post(fragments, uploaded_image, tone):
    """Generate Instagram/XHS post from text and optional image"""
    
    if not fragments.strip():
        return "❌ Please enter some text.", None
    
    try:
        # Step 0: Upload image if provided
        image_path = None
        if uploaded_image is not None:
            with open(uploaded_image, 'rb') as f:
                files = {'file': (os.path.basename(uploaded_image), f, 'image/png')}
                upload_resp = requests.post(
                    f"{API_URL}/upload_image",
                    files=files,
                    timeout=60
                )
                upload_resp.raise_for_status()
                image_path = upload_resp.json().get("image_path")
                print(f"Image uploaded: {image_path}")

        # Step 1: Intent extraction
        intent_resp = requests.post(
            f"{API_URL}/intent",
            json={
                "fragments": fragments,
                "image_path": image_path
            },
            timeout=TIMEOUT
        )
        intent_resp.raise_for_status()
        intent_data = intent_resp.json()

        intent = intent_data.get("intent", "")
        slots = intent_data.get("slots", {})
        slots["tone"] = tone  # Apply user's tone preference

        # Step 2: Outline generation
        outline_resp = requests.post(
            f"{API_URL}/outline",
            json={
                "intent": intent,
                "slots": slots,
                "image_path": image_path
            },
            timeout=TIMEOUT
        )
        outline_resp.raise_for_status()
        outline_data = outline_resp.json()

        outline = outline_data.get("outline", [])

        # Step 3: Content generation
        gen_resp = requests.post(
            f"{API_URL}/generate",
            json={
                "outline": outline,
                "tone": tone,
                "image_path": image_path
            },
            timeout=TIMEOUT
        )
        gen_resp.raise_for_status()
        gen_data = gen_resp.json()

        text = gen_data.get("text", "No text generated.")
        flagged = gen_data.get("flagged", False)

        # Step 4: Generate illustration
        img_path = None
        try:
            img_resp = requests.post(
                f"{API_URL}/image",
                json={"theme": intent},
                timeout=TIMEOUT
            )
            img_resp.raise_for_status()
            
            img_url = img_resp.json().get("image_url", None)
            
            if img_url:
                # Convert URL path to local file path
                img_path = f".{img_url}"
                
                # Verify file exists
                if not os.path.exists(img_path):
                    print(f"Warning: Image not found at {img_path}")
                    img_path = None
                else:
                    print(f"Generated image: {img_path}")
                    
        except Exception as e:
            print(f"Image generation error: {e}")

        # Format output
        status = "⚠️ Content was revised for safety" if flagged else "✅ Content is safe"
        final_text = f"**{status}**\n\n{text}"

        return final_text, img_path

    except requests.exceptions.Timeout:
        return "⏱️ Request timed out. The model might be processing, please try again.", None
    except requests.exceptions.ConnectionError:
        return "🔌 Cannot connect to backend. Make sure the server is running at port 8004.", None
    except requests.exceptions.HTTPError as e:
        return f"❌ Server error: {e.response.status_code}\n{e.response.text}", None
    except Exception as e:
        return f"❌ Error: {str(e)}", None


# ========== Gradio Interface ==========

with gr.Blocks(title="AI Mental Health Content Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 AI Mental Health Post Generator")
    gr.Markdown(
        "Generate professional mental health content for Instagram/XHS. "
        "Provide text description and optionally upload an image for context."
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Content Idea",
                placeholder="例如：最近很焦虑，睡不好，给大学生的温暖心理科普",
                lines=4,
                info="Describe what you want to create"
            )
            
            image_input = gr.Image(
                label="Upload Image (Optional)",
                type="filepath",
            )
            
            tone_select = gr.Dropdown(
                choices=["温和且专业", "学术科普", "轻松友好", "温柔引导"],
                value="温和且专业",
                label="Writing Style"
            )
            
            generate_btn = gr.Button("✨ Generate Post", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Tips")
            gr.Markdown("""
            **For best results:**
            - Be specific about your topic
            - Mention target audience
            - Optionally upload a relevant image
            
            **Examples:**
            - "压力管理技巧，面向职场人士"
            - "如何改善睡眠质量，温和的语气"
            - "焦虑情绪的应对方法，给大学生"
            """)

    with gr.Row():
        output_text = gr.Textbox(
            label="Generated Post",
            lines=18,
        )
        output_image = gr.Image(
            label="Generated Illustration",
            type="filepath"
        )

    generate_btn.click(
        fn=generate_post,
        inputs=[text_input, image_input, tone_select],
        outputs=[output_text, output_image]
    )

    gr.Markdown("""
    ---
    ### About
    This system uses **Qwen3-VL** for multimodal content understanding and generation, 
    plus **Stable Diffusion XL** for illustration generation. All content undergoes 
    automated safety review to ensure appropriate mental health messaging.
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=["./outputs"],
        show_error=True
    )