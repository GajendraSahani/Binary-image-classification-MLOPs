import logging
import time
import io
import os
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

metrics = {"request_count": 0}

MODEL_PATH = os.path.join("models", "model.h5")
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully.")
    else:
        logger.warning("Model file not found. Running without model.")


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded"}

    metrics["request_count"] += 1
    start_time = time.time()

    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert('RGB').resize((224, 224))
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)

    prediction = model.predict(img_array)
    label = "Dog" if prediction[0] > 0.5 else "Cat"

    latency = time.time() - start_time

    logger.info(
        f"Request #{metrics['request_count']} | "
        f"Label: {label} | "
        f"Latency: {latency:.4f}s"
    )

    return {
        "label": label,
        "confidence": float(prediction[0]),
        "latency": f"{latency:.4f}s",
        "total_requests_served": metrics["request_count"]
    }
 
