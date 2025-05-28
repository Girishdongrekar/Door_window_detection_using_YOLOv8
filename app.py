import gradio as gr
from ultralytics import YOLO
import json
import numpy as np

# Load YOLOv8 model
model = YOLO("best.pt")  # Replace with your model path or model name if on HF Hub

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

# Gradio UI
interface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[
        gr.Image(type="numpy", label="Detected Image"),
        gr.Textbox(label="Detection JSON")
    ],
    title="YOLOv8 Object Detection",
    description="Upload an image to detect objects. The output includes an annotated image and JSON with class_id, class_label, and bounding_box."
)

interface.launch()
