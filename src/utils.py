import numpy as np
from PIL import Image

def load_and_preprocess_image(file, img_size=224):
    image = Image.open(file).convert("RGB")
    image = image.resize((img_size, img_size))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image
