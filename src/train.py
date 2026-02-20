import tensorflow as tf
import mlflow
import yaml
import os

def build_model(img_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(img_size, img_size, 3)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def train():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    img_size = params["data"]["img_size"]
    epochs = params["training"]["epochs"]
    batch_size = params["training"]["batch_size"]

    train_ds = tf.keras.utils.image_dataset_from_directory(
        "data/raw",
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(img_size, img_size),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        "data/raw",
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(img_size, img_size),
        batch_size=batch_size
    )

    model = build_model(img_size)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    mlflow.start_run()

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    os.makedirs("models", exist_ok=True)
    model.save("models/model.h5")

    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_metric("final_accuracy", history.history["val_accuracy"][-1])

    mlflow.end_run()

if __name__ == "__main__":
    train()
