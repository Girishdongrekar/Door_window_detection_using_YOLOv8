# Door_window_detection_using_YOLOv8
This project focuses on automating the detection of doors and windows in architectural blueprint-style images using a custom-trained YOLOv8 object detection model.

🧠 What It Does:
Users can upload any blueprint image.

The trained model detects and highlights doors and windows.

It returns:

An annotated image with bounding boxes.

And also prints output containing class_id, class_label, and bounding_box coordinates.

🛠️ How It Works:
Images were labeled using Roboflow, which also helped preprocess and export the dataset in YOLO format.

The model was trained on these labeled images using YOLOv8 from Ultralytics.

A Gradio app was built (app.py) to allow easy image uploads and visualize predictions.

During testing, unlabeled blueprint images were used to validate the model's performance on real-world, unseen data.

📦 Project Structure:
dataset/ – Training images

classes.txt – Class names: door, window

app.py – Gradio-based object detection API

best.pt – Trained YOLOv8 model weights

README.md – Setup guide + curl test instructions

Screenshots – Labeling in Roboflow 

training_logs -  training logs/loss graph

Public API URL.txt - contains the link where the model is deployes and allowing anyone to test detection directly in the browser

🌐 Deployment:
The solution is deployed on Hugging Face Spaces, allowing anyone to test detection directly in the browser.
