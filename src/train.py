#Train

import tensorflow as tf
import mlflow
import mlflow.keras
from preprocess import preprocess_image

def build_model():
  model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224,224,3)),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(1, activation='sigmoid')
  ])
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  return model


if __name__ == '__main__':
  mlflow.set_experiment("Cats_vs_Dogs_Baseline")
  with mlflow.start_run():
    model = build_model()
    model.save("models/model.h5")
    mlflow.log_param("input_shape", (224,224,3))
    mlflow.keras.log_model(model, "model_artifact")
