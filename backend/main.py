from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from src.model import create_model
from src.transforms import get_val_transform
from backend.gradcam import generate_gradcam
import base64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
transform = None
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

app = FastAPI(title="Skin Lesion Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Skin Lesion Classifier API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.on_event("startup")
async def load_model():
    global model, transform
    
    model = create_model(num_classes=7)
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    model.to(device)
    model.eval()

    transform = get_val_transform()
    
    print("Model loaded successfully")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
    
    # Get top prediction
    top_class_idx = probabilities.argmax().item()
    
    # Generate Grad-CAM for better interpretability
    # (we need gradients for Grad-CAM)
    input_tensor.requires_grad = True
    cam = generate_gradcam(model, input_tensor, top_class_idx)
    
    # add heatmap overlay and convert to base64 for JSON response
    cam_image = create_heatmap_overlay(image, cam)
    
    predictions = [
        {"class": class_names[i], "probability": float(probabilities[i])}
        for i in range(len(class_names))
    ]
    predictions.sort(key=lambda x: x['probability'], reverse=True)
    
    return {
        "predictions": predictions,
        "top_prediction": predictions[0]["class"],
        "confidence": predictions[0]["probability"],
        "gradcam": cam_image  # encoded heatmap (base64)
    }


def create_heatmap_overlay(original_image, cam):
    """
    Creates a heatmap overlay on the original image
    
    Returns: base64 encoded PNG
    """
    img_resized = original_image.resize((224, 224))
    img_array = np.array(img_resized)
    
    # Apply colormap to CAM
    heatmap = cm.jet(cam)[:, :, :3]  # RGB only, no alpha
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # 60% original, 40% heatmap
    overlay = (0.6 * img_array + 0.4 * heatmap).astype(np.uint8)
    
    overlay_img = Image.fromarray(overlay)
    buffered = io.BytesIO()
    overlay_img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    return f"data:image/png;base64,{img_base64}"