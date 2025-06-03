import gradio as gr
from ultralytics import YOLO
import json
import numpy as np

# Load YOLOv8 model
model = YOLO("best.pt")  # Replace with your actual path or HF model name

def detect_objects(image):
    results = model(image)
    result = results[0]

    # Annotated image
    annotated_image = result.plot()

    # Convert detection info to JSON
    detection_list = []
    for box in result.boxes:
        xyxy = box.xyxy[0].tolist()
        class_id = int(box.cls[0])
        class_label = model.names[class_id]
        detection_list.append({
            "class_id": class_id,
            "class_label": class_label,
            "bounding_box": [round(coord, 2) for coord in xyxy]
        })

    detection_json = json.dumps(detection_list, indent=2)

    return annotated_image, detection_json

# Gradio Instagram-style UI
with gr.Blocks(theme=gr.themes.Soft(), title="YOLOv8 Object Detector") as demo:
    gr.Markdown("""
    <div style="text-align: center;">
        <h1 style="font-size: 2.5rem; font-family: 'Arial', sans-serif;">📷 YOLOv8 Vision</h1>
        <p style="font-size: 1.1rem;">Upload a photo and let the AI tag the scene like magic ✨</p>
    </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="📤 Upload Your Snap", image_mode="RGB")
            submit_btn = gr.Button("✨ Detect Objects", variant="primary")
        
        with gr.Column(scale=1):
            output_image = gr.Image(type="numpy", label="📍 Tagged Image", image_mode="RGB", show_label=True)
            output_json = gr.Textbox(label="🧾 Detection Details (JSON)", lines=12, show_copy_button=True)

    submit_btn.click(fn=detect_objects, inputs=input_image, outputs=[output_image, output_json])

    gr.Markdown("""
    <div style="text-align: center; font-size: 0.9rem; color: gray; margin-top: 20px;">
        💡 Tip: Upload selfies, pets, or scenery—YOLOv8 will label them!<br/>
        🚀 Built with YOLOv8 & Gradio for an Insta-friendly AI vibe.
    </div>
    """)

demo.launch()
