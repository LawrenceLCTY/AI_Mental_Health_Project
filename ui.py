# ui.py
import gradio as gr
import requests

API_URL = "http://127.0.0.1:8004"

def generate_post(fragments):
    if not fragments.strip():
        return "❌ Please enter some text.", None

    # 1. Intent
    intent_resp = requests.post(
        f"{API_URL}/intent",
        json={"fragments": fragments}
    ).json()

    intent = intent_resp.get("intent", "")
    slots = intent_resp.get("slots", {})

    # 2. Outline
    outline_resp = requests.post(
        f"{API_URL}/outline",
        json={"intent": intent, "slots": slots}
    ).json()

    outline = outline_resp.get("outline", [])

    # 3. Generate text
    gen_resp = requests.post(
        f"{API_URL}/generate",
        json={
            "outline": outline,
            "tone": slots.get("tone", "温和")
        }
    ).json()

    text = gen_resp.get("text", "No text generated.")
    flagged = gen_resp.get("flagged", False)

    # 4. Image
    img_path = None
    try:
        img_resp = requests.post(
            f"{API_URL}/image",
            json={"theme": intent}
        ).json()
        
        img_url = img_resp.get("image_url", None)
        
        if img_url:
            # Convert URL path to local file path
            # From: /outputs/images/mental_health_...png
            # To: ./outputs/images/mental_health_...png
            img_path = f".{img_url}"  # Just add a dot at the beginning
            print(f"Image path: {img_path}")
            
    except Exception as e:
        print(f"Image error: {e}")
        import traceback
        traceback.print_exc()

    status = "⚠️ Safety review applied" if flagged else "✅ Safe"

    print(f"Final img_path being returned: {img_path}")
    return f"{status}\n\n{text}", img_path



with gr.Blocks(title="AI Mental Health Content Generator") as demo:
    gr.Markdown("# 🧠 AI Mental Health Content Generator")
    gr.Markdown(
        "Enter **short phrases or ideas**. The system will turn them into a complete, safe mental-health post."
    )

    with gr.Row():
        input_box = gr.Textbox(
            label="Your idea / short phrases",
            placeholder="例如：最近很焦虑，睡不好，给大学生的温暖心理科普",
            lines=3
        )

    generate_btn = gr.Button("✨ Generate Post")

    with gr.Row():
        output_text = gr.Textbox(
            label="Generated Post",
            lines=15
        )

    output_image = gr.Image(label="Generated Image", type="filepath")

    generate_btn.click(
        fn=generate_post,
        inputs=input_box,
        outputs=[output_text, output_image]
    )

demo.launch(server_name="0.0.0.0", 
            server_port=7860, 
            allowed_paths=["./outputs"])