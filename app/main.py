import logging
import time
import io
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename="app_monitoring.log"
)
logger = logging.getLogger("InferenceService")

app = FastAPI()
model = tf.keras.models.load_model("models/model.h5")

metrics = {"request_count": 0}

@app.get("/health") # M2 Task 1 health check [cite: 24]
async def health():
    return {"status": "healthy"}

@app.post("/predict") # M2 Task 1 prediction endpoint [cite: 24]
async def predict(file: UploadFile = File(...)):
    metrics["request_count"] += 1
    start_time = time.time()
    
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert('RGB').resize((224, 224))
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    
    # Inference
    prediction = model.predict(img_array)
    label = "Dog" if prediction[0] > 0.5 else "Cat"
    
    # M5 Task 1: Track latency 
    latency = time.time() - start_time
    
    logger.info(f"Request #{metrics['request_count']} | Label: {label} | Latency: {latency:.4f}s")
    
    return {
        "label": label, 
        "confidence": float(prediction[0]),
        "latency": f"{latency:.4f}s",
        "total_requests_served": metrics["request_count"]
    }
