from pathlib import Path
from typing import Final
from ultralytics import YOLO

import argparse

# Create a command line argument processor for model and dataset
parser = argparse.ArgumentParser(description="YOLO Training and Testing")
parser.add_argument("--model", "-m", type=str, default="YOLO11m", help="Model type")
parser.add_argument("--dataset", "-d", type=str, default="POP", help="Dataset used")

ARGUMENTS: Final[argparse.Namespace] = parser.parse_args()

# Model used
MODEL_TYPE: Final[str] = ARGUMENTS.model

# Dataset used
DATASET_USED: Final[str] = ARGUMENTS.dataset

# General train parameters
EPOCHS: Final[int] = 1000
PATIENCE: Final[int] = 100
BATCH: Final[float] = 0.90
IMAGE_SIZE: Final[int] = 1024
SAVE: Final[bool] = True
CACHE: Final[bool] = True
WORKERS: Final[int] = 16
PROJECT: Final[str] = "YOLO"
NAME: Final[str] = f"{MODEL_TYPE.upper()}_{DATASET_USED.upper()}"
EXIST_OK: Final[bool] = True
PRETRAINED: Final[bool] = True
OPTIMIZER: Final[str] = "auto"
SEED: Final[int] = 42
DETERMINISTIC: Final[bool] = False
PLOTS: Final[bool] = True
AMP: Final[bool] = False

# General test parameters
VAL_IMAGE_SIZE: Final[int] = IMAGE_SIZE
VAL_BATCH: Final[int] = 64
VAL_SAVE_JSON: Final[bool] = True
VAL_MAX_DET: Final[int] = 50
VAL_PLOTS: Final[bool] = True
VAL_PROJECT: Final[str] = PROJECT
VAL_NAME: Final[str] = NAME


MODEL_NAME: Final[str] = f"{MODEL_TYPE.lower()}.pt"
DATASETS_PATH: Final[Path] = Path("datasets")
DATASET_VAL_YAML: Final[Path] = DATASETS_PATH / DATASET_USED / "data_val.yaml"
DATASET_TEST_YAML: Final[Path] = DATASETS_PATH / DATASET_USED / "data_test.yaml"


def main() -> None:
    # Load the model
    model: Final[YOLO] = YOLO(MODEL_NAME)

    # Train the model
    results = model.train(
        data=DATASET_VAL_YAML,
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

    # Test the model
    metrics = model.val(
        data=DATASET_TEST_YAML,
        imgsz=VAL_IMAGE_SIZE,
        batch=VAL_BATCH,
        save_json=VAL_SAVE_JSON,
        max_det=VAL_MAX_DET,
        plots=VAL_PLOTS,
        project=VAL_PROJECT,
        name=VAL_NAME
    )

    # Save the metrics to a file
    with open(f"{PROJECT}/{NAME}/metrics.txt", "w", encoding="UTF-8") as file:
        file.write(str(metrics))


if __name__ == "__main__":
    main()
