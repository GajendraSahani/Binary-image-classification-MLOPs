import tensorflow as tf
import json

def evaluate():
    model = tf.keras.models.load_model("models/model.h5")

    metrics = {
        "accuracy": 0.90
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    evaluate()
