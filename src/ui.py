"""
Gradio UI for Bone Fracture Classification.
Premium Glassmorphism Interface.
"""
from pathlib import Path
import gradio as gr
import numpy as np

from src.config import CLASS_NAMES, MODEL_REGISTRY
from src.inference import predict

def create_ui(models_dict: dict) -> gr.Blocks:
    """Builds the Gradio UI using the provided loaded models."""
    
    def classify_xray(image, model_choice):
        empty_html = '<div class="verdict-box verdict-empty">Awaiting Scan...</div>'
        if image is None:
            return {}, empty_html

        model_key = "cnn" if "CNN" in model_choice else "mobilenetv3"
        if model_key not in models_dict:
            return {}, f'<div class="verdict-box verdict-fractured">❌ Model Not Loaded</div>'
        
        try:
            result = predict(models_dict[model_key], image, model_name=model_key)
            probs = result["probabilities"]
            confidence = result["confidence"]
            label = result["prediction"]
            
            if label == "fractured":
                html = f'<div class="verdict-box verdict-fractured">🦴 FRACTURED<br><span style="font-size:0.7em;font-weight:500;opacity:0.8;">Diagnostic Confidence: {confidence:.1%}</span></div>'
            else:
                html = f'<div class="verdict-box verdict-normal">✅ NORMAL BONE<br><span style="font-size:0.7em;font-weight:500;opacity:0.8;">Diagnostic Confidence: {confidence:.1%}</span></div>'
            return probs, html
        except Exception as e:
            return {}, f'<div class="verdict-box verdict-fractured">❌ Error: {str(e)}</div>'

    SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
    sample_images = sorted(SAMPLES_DIR.glob("*.png")) + sorted(SAMPLES_DIR.glob("*.jpg")) + sorted(SAMPLES_DIR.glob("*.jpeg"))
    sample_paths = [str(p) for p in sample_images[:10]]

    CUSTOM_CSS = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&display=swap');
    
    body { font-family: 'Inter', sans-serif !important; background-color: #f8fafc !important; }
    
    .gradio-container { 
        max-width: 1050px !important; 
        margin: 40px auto !important; 
        background: rgba(255, 255, 255, 0.6) !important; 
        backdrop-filter: blur(25px) !important; 
        -webkit-backdrop-filter: blur(25px) !important;
        border-radius: 24px !important; 
        border: 1px solid rgba(255, 255, 255, 0.8) !important; 
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.08) !important; 
        padding: 40px !important; 
    }
    
    h1 {
        text-align: center;
        background: linear-gradient(135deg, #4f46e5 0%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.2em !important;
        font-weight: 800 !important;
        margin-bottom: 5px !important;
        letter-spacing: -0.03em;
    }
    
    .subtitle { 
        text-align: center; color: #64748b; font-size: 1.15em; font-weight: 500; margin-bottom: 35px; 
    }
    
    .card-panel {
        background: #ffffff !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.04) !important;
        padding: 25px !important;
        border: 1px solid #f1f5f9 !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card-panel:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.08) !important;
    }
    
    button.primary {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        border: none !important;
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.3) !important;
        transition: all 0.3s ease !important;
        font-weight: 600 !important;
        font-size: 1.1em !important;
        border-radius: 12px !important;
        padding: 12px !important;
    }
    button.primary:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 25px rgba(79, 70, 229, 0.45) !important;
    }
    
    .verdict-box {
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        font-size: 1.8em;
        font-weight: 800;
        margin-top: 10px;
        transition: all 0.4s ease;
        letter-spacing: -0.02em;
    }
    .verdict-fractured {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(220, 38, 38, 0.05));
        color: #dc2626;
        border: 2px solid rgba(239, 68, 68, 0.4);
        box-shadow: 0 10px 25px rgba(239, 68, 68, 0.15);
    }
    .verdict-normal {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(22, 163, 74, 0.05));
        color: #16a34a;
        border: 2px solid rgba(34, 197, 94, 0.4);
        box-shadow: 0 10px 25px rgba(34, 197, 94, 0.15);
    }
    .verdict-empty {
        background: #f8fafc;
        color: #94a3b8;
        border: 2px dashed #cbd5e1;
        font-size: 1.2em;
        padding: 40px 20px;
    }
    
    /* Make the gallery container look like a medical film strip */
    .gallery-container {
        margin-top: 40px !important;
        padding: 30px !important;
        background: #1e293b !important;
        border-radius: 20px !important;
        border: 4px solid #0f172a !important;
        box-shadow: inset 0 4px 20px rgba(0,0,0,0.5) !important;
    }
    .gallery-container h3 {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        margin-top: 0 !important;
    }
    """

    model_choices = [entry['name'] for key, entry in MODEL_REGISTRY.items()]

    # We use a custom theme configuration for softer inputs
    theme = gr.themes.Default(
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        primary_hue="indigo",
        secondary_hue="pink",
        neutral_hue="slate",
    ).set(
        block_background_fill="transparent",
        block_border_width="0px",
    )

    with gr.Blocks(css=CUSTOM_CSS, title="AI Radiologist | Bone Fracture", theme=theme) as demo:
        gr.Markdown("<h1>🦴 AI Radiologist</h1>")
        gr.Markdown('<p class="subtitle">State-of-the-art Deep Learning analysis for real-time Bone Fracture detection.</p>')

        with gr.Row():
            # Left Column (Input)
            with gr.Column(scale=1, elem_classes="card-panel"):
                image_input = gr.Image(type="pil", label="Upload X-Ray Image", height=380)
                
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=model_choices,
                        value=model_choices[-1] if model_choices else None,
                        label="Select Neural Network Architecture",
                        container=False
                    )
                
                predict_btn = gr.Button("🔍 Run AI Diagnostics", variant="primary", size="lg")

            # Right Column (Output)
            with gr.Column(scale=1, elem_classes="card-panel"):
                verdict_output = gr.HTML(value='<div class="verdict-box verdict-empty">Awaiting Scan...</div>')
                
                gr.Markdown("<br>**Neural Network Confidence Distribution**")
                label_output = gr.Label(label="", num_top_classes=2, show_label=False)

        # Bottom Gallery
        if sample_paths:
            with gr.Column(elem_classes="gallery-container"):
                gr.Markdown("### 📸 Quick Test Gallery (Click an X-Ray)")
                gr.Examples(
                    examples=sample_paths, 
                    inputs=image_input, 
                    examples_per_page=10, 
                    label=""
                )
        else:
            with gr.Column(elem_classes="gallery-container"):
                gr.Markdown("### 📸 Quick Test Gallery\n<span style='color:#94a3b8'>*Upload images to the `samples/` folder to populate this film strip.*</span>")

        # Events
        predict_btn.click(fn=classify_xray, inputs=[image_input, model_dropdown], outputs=[label_output, verdict_output])
        image_input.change(fn=classify_xray, inputs=[image_input, model_dropdown], outputs=[label_output, verdict_output])

    return demo
