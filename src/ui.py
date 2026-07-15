"""
Gradio UI for Bone Fracture Classification.
Professional Medical Enterprise Interface.
"""
from pathlib import Path
import gradio as gr
import numpy as np

from src.config import CLASS_NAMES, MODEL_REGISTRY
from src.inference import predict

def create_ui(models_dict: dict) -> gr.Blocks:
    """Builds the Gradio UI using the provided loaded models."""
    
    def classify_xray(image, model_choice):
        empty_html = '<div class="verdict-box verdict-empty">Awaiting Radiograph...</div>'
        if image is None:
            return {}, empty_html

        model_key = "cnn" if "CNN" in model_choice else "mobilenetv3"
        if model_key not in models_dict:
            return {}, f'<div class="verdict-box verdict-error">System Error: Model Not Loaded</div>'
        
        try:
            result = predict(models_dict[model_key], image, model_name=model_key)
            probs = result["probabilities"]
            confidence = result["confidence"]
            label = result["prediction"]
            
            if label == "fractured":
                html = f'<div class="verdict-box verdict-fractured">FRACTURE DETECTED<br><span class="verdict-subtext">Confidence Score: {confidence:.1%}</span></div>'
            else:
                html = f'<div class="verdict-box verdict-normal">NO FRACTURE DETECTED<br><span class="verdict-subtext">Confidence Score: {confidence:.1%}</span></div>'
            return probs, html
        except Exception as e:
            return {}, f'<div class="verdict-box verdict-error">Processing Error: {str(e)}</div>'

    SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
    sample_images = sorted(SAMPLES_DIR.glob("*.png")) + sorted(SAMPLES_DIR.glob("*.jpg")) + sorted(SAMPLES_DIR.glob("*.jpeg"))
    # Gradio 4.x requires a list of lists for examples!
    sample_paths = [[str(p)] for p in sample_images[:10]]

    CUSTOM_CSS = """
    body { background-color: #f4f7f6 !important; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important; }
    
    .gradio-container { 
        max-width: 1100px !important; 
        margin: 30px auto !important; 
        background: #ffffff !important; 
        border-radius: 8px !important; 
        border: 1px solid #e2e8f0 !important; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important; 
        padding: 30px !important; 
    }
    
    .header-container {
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 15px;
        margin-bottom: 25px;
    }
    
    h1 {
        color: #1e293b !important;
        font-size: 24px !important;
        font-weight: 600 !important;
        margin: 0 !important;
        letter-spacing: 0.5px;
    }
    
    .subtitle { 
        color: #64748b; 
        font-size: 14px; 
        margin-top: 4px; 
    }
    
    .panel {
        background: #f8fafc !important;
        border-radius: 6px !important;
        padding: 20px !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    button.primary {
        background-color: #0f172a !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 4px !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        padding: 10px 16px !important;
        transition: background-color 0.2s ease !important;
    }
    button.primary:hover {
        background-color: #334155 !important;
    }
    
    .verdict-box {
        padding: 15px;
        border-radius: 4px;
        text-align: center;
        font-size: 18px;
        font-weight: 600;
        margin-top: 10px;
        border-width: 1px;
        border-style: solid;
    }
    .verdict-fractured {
        background-color: #fef2f2;
        color: #991b1b;
        border-color: #f87171;
    }
    .verdict-normal {
        background-color: #f0fdf4;
        color: #166534;
        border-color: #4ade80;
    }
    .verdict-empty {
        background-color: #f1f5f9;
        color: #64748b;
        border-color: #cbd5e1;
    }
    .verdict-error {
        background-color: #fffbeb;
        color: #92400e;
        border-color: #fbbf24;
    }
    .verdict-subtext {
        font-size: 12px;
        font-weight: 400;
        color: inherit;
        opacity: 0.8;
    }
    
    .gallery-container {
        margin-top: 30px !important;
        padding: 20px !important;
        background: #ffffff !important;
        border-top: 1px solid #e2e8f0 !important;
    }
    .gallery-container h3 {
        color: #334155 !important;
        font-size: 16px !important;
        margin-bottom: 10px !important;
        font-weight: 600 !important;
    }
    
    .prototype-warning {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fcd34d;
        border-radius: 4px;
        padding: 12px 15px;
        font-size: 13px;
        font-weight: 500;
        text-align: center;
        margin-top: 20px;
    }
    """

    model_choices = [entry['name'] for key, entry in MODEL_REGISTRY.items()]

    with gr.Blocks(css=CUSTOM_CSS, title="Bone Fracture Detection System") as demo:
        with gr.Column(elem_classes="header-container"):
            gr.Markdown("<h1>Bone Fracture Detection System</h1>")
            gr.Markdown('<div class="subtitle">Clinical Decision Support Tool - MURA Dataset Trained Models</div>')
            gr.Markdown('<div class="prototype-warning"><strong>PROTOTYPE / RESEARCH USE ONLY:</strong> This system is a prototype and is not intended for clinical diagnostics. All predictions must be verified by a qualified medical professional.</div>')

        with gr.Row():
            # Left Column (Input)
            with gr.Column(scale=1, elem_classes="panel"):
                image_input = gr.Image(type="pil", label="Radiograph Input", height=350)
                
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=model_choices,
                        value=model_choices[-1] if model_choices else None,
                        label="Architecture Selection",
                        container=False
                    )
                
                predict_btn = gr.Button("Analyze Radiograph", variant="primary")

            # Right Column (Output)
            with gr.Column(scale=1, elem_classes="panel"):
                verdict_output = gr.HTML(value='<div class="verdict-box verdict-empty">Awaiting Radiograph...</div>')
                
                gr.Markdown("<br>**Model Confidence Distribution**")
                label_output = gr.Label(label="", num_top_classes=2, show_label=False)

        # Bottom Gallery
        with gr.Column(elem_classes="gallery-container"):
            gr.Markdown("### Selected Patient Examples")
            if sample_paths:
                gr.Examples(
                    examples=sample_paths, 
                    inputs=image_input, 
                    examples_per_page=10, 
                    label=""
                )
            else:
                gr.Markdown("<span style='color:#64748b; font-size: 14px;'>No reference examples found in the designated samples directory.</span>")

        # Events
        predict_btn.click(fn=classify_xray, inputs=[image_input, model_dropdown], outputs=[label_output, verdict_output])
        image_input.change(fn=classify_xray, inputs=[image_input, model_dropdown], outputs=[label_output, verdict_output])

    return demo
