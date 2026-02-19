import numpy as np
from PIL import Image

def preprocess_image(image_path):
  img = Image.open(image_path).convert('RGB')
  img = img.resize((224, 224))
  img_array = np.array(img) / 255.0
  return img_array
