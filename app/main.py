import time
import logging
import numpy as np
from fastapi import FastAPI, UploadFile, File
from src.utils import load_and_preprocess_image
import tensorflow as tf

app = FastAPI()

logging.basicConfig(level=logging.INFO)

model = None
request_count = 0

@app.on_event("startup")
def load_model():
    global model
    model = tf.keras.models.load_model("models/model.h5")
    logging.info("Model loaded successfully")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "requests_served": request_count
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global request_count

    start_time = time.time()

    image = load_and_preprocess_image(file.file)

    prediction = model.predict(image)[0][0]
    label = "Dog" if prediction > 0.5 else "Cat"

    latency = time.time() - start_time
    request_count += 1

    logging.info(f"Prediction latency: {latency}")

    return {
        "label": label,
        "confidence": float(prediction),
        "latency": latency
    }
