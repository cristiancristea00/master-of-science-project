from os import SEEK_DATA
from pathlib import Path
from typing import Final
from ultralytics import YOLO

# Model used
MODEL_TYPE: Final[str] = "YOLO11n"

# Dataset used
DATASET_USED: Final[str] = "POP"

# General parameters
EPOCHS: Final[int] = 1000
PATIENCE: Final[int] = 50
BATCH: Final[int] = -1
IMAGE_SIZE: Final[int] = 1024
SAVE: Final[bool] = True
CACHE: Final[bool] = True
WORKERS: Final[int] = 16
PROJECT: Final[str] = "YOLO11"
EXIST_OK: Final[bool] = True
PRETRAINED: Final[bool] = True
OPTIMIZER: Final[str] = "auto"
SEED: Final[int] = 42
DETERMINISTIC: Final[bool] = True
PLOTS: Final[bool] = True
AMP: Final[bool] = False


MODEL_NAME: Final[str] = f"{MODEL_TYPE.lower()}.pt"

DATASETS_PATH: Final[Path] = Path("datasets")
DATASET_YAML: Final[Path] = DATASETS_PATH / DATASET_USED / "data.yaml"


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
    exist_ok=EXIST_OK,
    pretrained=PRETRAINED,
    optimizer=OPTIMIZER,
    seed=SEED,
    deterministic=DETERMINISTIC,
    plots=PLOTS,
    amp=AMP
)
