from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import io
import asyncio

import numpy as np
from pathlib import Path

from src.model import create_model
from src.transforms import get_val_transform
from backend.gradcam import generate_gradcam
import base64

device = 'cpu'
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

PROJECT_ROOT = Path(__file__).parent.parent  # backend/main.py -> project root
MODEL_PATH = PROJECT_ROOT / 'models' / 'best_model.pth'

@app.get("/")
async def root():
    return {"message": "Skin Lesion Classifier API", "status": "running"}

@app.on_event("startup")
async def load_model():
    global model, transform
    
    model = create_model(num_classes=7)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    transform = get_val_transform()
    
    print("Model loaded successfully!")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device),
        "classes": class_names
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    #check if image
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail=f"File must be an image. Got: {file.content_type}"
        )
    
    # max 10MB
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:  # 10MB in bytes
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 10MB"
        )
    
    try:
        # Try to open image
        image = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {str(e)}"
        )
    
    try:
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
        
        top_class_idx = probabilities.argmax().item()
        
        # GradCAM for improved interpretability
        input_tensor.requires_grad = True
        cam = generate_gradcam(model, input_tensor, top_class_idx)
        cam_image = create_heatmap_overlay(image, cam)
        
        # Format and convert to base64 for JSON
        predictions = [
            {"class": class_names[i], "probability": float(probabilities[i])}
            for i in range(len(class_names))
        ]
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        return {
            "predictions": predictions,
            "top_prediction": predictions[0]["class"],
            "confidence": predictions[0]["probability"],
            "gradcam": cam_image
        }

    # catch other errors
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


def create_heatmap_overlay(original_image, cam):
    """
    Creates a heatmap overlay on the original image
    
    Returns: base64 encoded PNG
    """
    img_resized = original_image.resize((224, 224))
    img_array = np.array(img_resized)
    
    r = np.clip(np.minimum(4 * cam - 1.5, -4 * cam + 4.5), 0, 1)
    g = np.clip(np.minimum(4 * cam - 0.5, -4 * cam + 3.5), 0, 1)
    b = np.clip(np.minimum(4 * cam + 0.5, -4 * cam + 2.5), 0, 1)
    
    heatmap = np.stack([r, g, b], axis=-1)
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # 60% original, 40% heatmap
    overlay = (0.6 * img_array + 0.4 * heatmap).astype(np.uint8)
    
    overlay_img = Image.fromarray(overlay)
    buffered = io.BytesIO()
    overlay_img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    return f"data:image/png;base64,{img_base64}"

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Catch any unhandled exceptions"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )

@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=30.0)
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={"error": "Request timeout", "detail": "Processing took too long"}
        )