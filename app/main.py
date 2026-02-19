import logging
import time
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MLOps-Inference")

app = FastAPI()
model = tf.keras.models.load_model("models/model.h5")

@app.get("/health") [cite: 24]
async def health():
    return {"status": "healthy"}

@app.post("/predict") [cite: 24]
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert('RGB').resize((224, 224))
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    
    prediction = model.predict(img_array)
    label = "Dog" if prediction[0] > 0.5 else "Cat"
    
    latency = time.time() - start_time
    logger.info(f"Label: {label} | Latency: {latency:.4f}s") [cite: 58]
    
    return {"label": label, "latency": latency}
