from pathlib import Path
from typing import Final
from ultralytics import YOLO

import argparse

YOLO_V8_MODELS: Final[set] = { "YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x" }
YOLO_V11_MODELS: Final[set] = { "YOLO11n", "YOLO11s", "YOLO11m", "YOLO11l", "YOLO11x" }
YOLO_V12_MODELS: Final[set] = { "YOLO12n", "YOLO12s", "YOLO12m", "YOLO12l", "YOLO12x" }

YOLO_MODELS: Final[set] = YOLO_V8_MODELS | YOLO_V11_MODELS | YOLO_V12_MODELS

# Create a command line argument processor for model and dataset
parser = argparse.ArgumentParser(description="YOLO Training and Testing")
parser.add_argument("--model", "-m", type=str, default="YOLO12m", help="Model type", choices=YOLO_MODELS, required=True)
parser.add_argument("--dataset", "-d", type=str, default="POP", help="Dataset used", choices=("POP", "LLVIP"), required=True)
parser.add_argument("--spectrum", "-s", type=str, default="infrared", help="Spectrum used", choices=("infrared", "visible"))

ARGUMENTS: Final[argparse.Namespace] = parser.parse_args()

# Model used
MODEL_TYPE: Final[str] = ARGUMENTS.model

# Dataset used
DATASET_USED: Final[str] = ARGUMENTS.dataset

# Spectrum used
SPECTRUM: Final[str] = ARGUMENTS.spectrum

# General train parameters
EPOCHS: Final[int] = 1000
PATIENCE: Final[int] = 100
BATCH: Final[float] = 0.90
IMAGE_SIZE: Final[int] = 1024
SAVE: Final[bool] = True
CACHE: Final[bool] = True
WORKERS: Final[int] = 16
PROJECT: Final[str] = "YOLO"
NAME: Final[str] = f"{MODEL_TYPE.upper()}_{DATASET_USED.upper()}" if DATASET_USED != "LLVIP" else f"{MODEL_TYPE.upper()}_{DATASET_USED.upper()}_{SPECTRUM.upper()}"
EXIST_OK: Final[bool] = True
PRETRAINED: Final[bool] = True
OPTIMIZER: Final[str] = "auto"
SEED: Final[int] = 42
DETERMINISTIC: Final[bool] = False
PLOTS: Final[bool] = True
AMP: Final[bool] = False


MODEL_NAME: Final[str] = f"{MODEL_TYPE.lower()}.pt"
DATASETS_PATH: Final[Path] = Path("datasets")
DATASET_YAML: Final[Path] = DATASETS_PATH / DATASET_USED / "data.yaml" if DATASET_USED != "LLVIP" else DATASETS_PATH / DATASET_USED / f"data_{SPECTRUM}.yaml"


def main() -> None:
    # Load the model
    model: Final[YOLO] = YOLO(MODEL_NAME)

    # Train the model
    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        patience=PATIENCE,
        batch=BATCH,
        imgsz=IMAGE_SIZE,
        save=SAVE,
        cache=CACHE,
        workers=WORKERS,
        project=PROJECT,
        name=NAME,
        exist_ok=EXIST_OK,
        pretrained=PRETRAINED,
        optimizer=OPTIMIZER,
        seed=SEED,
        deterministic=DETERMINISTIC,
        plots=PLOTS,
        amp=AMP
    )

    # Save the results to a file
    with open(f"{PROJECT}/{NAME}/results.txt", "w", encoding="UTF-8") as file:
        file.write(str(results))


if __name__ == "__main__":
    main()
