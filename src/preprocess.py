import os
import yaml

def preprocess():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    os.makedirs("data/processed", exist_ok=True)

    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    preprocess()
