# Configuration variables
SBATCH_CMD = sbatch
TRAINING_SCRIPT = yolo.sh

# Default target
.PHONY: all
all: train

# Training target
.PHONY: train
train:
	@echo "Submitting YOLO training job..."
	$(SBATCH_CMD) $(TRAINING_SCRIPT) $(MODEL) $(DATASET)
	@echo "Job submitted. Check status with 'squeue -u cristian.cristea'."

# Cleaning target - removes only .err and .out files
.PHONY: clean
clean:
	@echo "Cleaning Slurm output files (.err and .out)..."
	rm -f *.err *.out
	@echo "Slurm output files cleaned."

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  train  - Submit YOLO training job to Slurm"
	@echo "  clean  - Remove Slurm output files (.err and .out)"
	@echo "  help   - Display this help message"
