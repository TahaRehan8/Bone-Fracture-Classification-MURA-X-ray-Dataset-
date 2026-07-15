"""
Gradio UI for Bone Fracture Classification.
Designed to be mounted directly into the FastAPI application.
"""
from pathlib import Path
import gradio as gr
import numpy as np

from src.config import CLASS_NAMES, MODEL_REGISTRY
from src.inference import predict

def create_ui(models_dict: dict) -> gr.Blocks:
    """Builds the Gradio UI using the provided loaded models."""
    
    def classify_xray(image, model_choice):
        if image is None:
            return {}, "⚠️ Please upload or select an image."

        model_key = "cnn" if "CNN" in model_choice else "mobilenetv3"
        if model_key not in models_dict:
            return {}, f"❌ Model '{model_key}' is not loaded."
        
        try:
            result = predict(models_dict[model_key], image, model_name=model_key)
            probs = result["probabilities"]
            confidence = result["confidence"]
            label = result["prediction"]
            
            if label == "fractured":
                emoji = "🦴🔴"
                verdict = f"{emoji}  **FRACTURED** — Confidence: {confidence:.1%}"
            else:
                emoji = "✅"
                verdict = f"{emoji}  **NOT FRACTURED** — Confidence: {confidence:.1%}"
            return probs, verdict
        except Exception as e:
            return {}, f"❌ Error: {str(e)}"

    SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
    sample_images = sorted(SAMPLES_DIR.glob("*.png")) + sorted(SAMPLES_DIR.glob("*.jpg")) + sorted(SAMPLES_DIR.glob("*.jpeg"))
    sample_paths = [str(p) for p in sample_images[:10]]

    CUSTOM_CSS = """
    .gradio-container { max-width: 960px !important; margin: auto; }
    h1 {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2em !important;
        margin-bottom: 0 !important;
    }
    .subtitle { text-align: center; color: #6b7280; font-size: 1.05em; margin-top: 4px; margin-bottom: 20px; }
    """

    model_choices = []
    for key, entry in MODEL_REGISTRY.items():
        # Just use names for choices since params will be dynamic
        model_choices.append(f"{entry['name']}")

    with gr.Blocks(css=CUSTOM_CSS, title="Bone Fracture Classifier", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🦴 Bone Fracture Classifier")
        gr.Markdown('<p class="subtitle">Upload an X-ray or select a sample image below to detect bone fractures using deep learning.</p>')

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="X-Ray Image", height=360)
                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    value=model_choices[-1] if model_choices else None,
                    label="🧠 Select Model",
                )
                predict_btn = gr.Button("🔍 Analyze X-Ray", variant="primary", size="lg")

            with gr.Column(scale=1):
                verdict_output = gr.Markdown(label="Verdict", value="*Upload an image and click Analyze*")
                label_output = gr.Label(label="Class Probabilities", num_top_classes=2)

        if sample_paths:
            gr.Markdown("---")
            gr.Markdown("### 📸 Sample X-Rays — Click to Test")
            gr.Examples(examples=sample_paths, inputs=image_input, examples_per_page=10, label="")
        else:
            gr.Markdown(
                "---\n### 📸 Sample X-Rays\n"
                "> **No sample images found.** Add `.png` or `.jpg` files to the `samples/` directory to enable the 10-photo quick select option.\n"
            )

        predict_btn.click(fn=classify_xray, inputs=[image_input, model_dropdown], outputs=[label_output, verdict_output])
        image_input.change(fn=classify_xray, inputs=[image_input, model_dropdown], outputs=[label_output, verdict_output])

    return demo
