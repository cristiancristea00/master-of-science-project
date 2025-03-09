#!/bin/bash

#SBATCH --partition=dgxh100           # Partition (queue) name
#SBATCH --cpus-per-task=34            # Number of CPU cores per task
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --mem=200G                    # Memory request (200 GB)
#SBATCH --job-name=YOLO               # Job name
#SBATCH --output=YOLO_%j.out          # Standard output log
#SBATCH --mail-type=BEGIN,END,FAIL    # Mail type
#SBATCH --mail-user=cristian.cristea@stud.etti.upb.ro


# Print job information
echo ""
echo "================ Job Information ================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Start time: $(date '+%A, %-d %B %Y, %I:%M:%S %p')"
echo "Working directory: $(pwd)"
echo "Node: $(hostname)"
echo "CPU cores: $SLURM_CPUS_PER_TASK"
echo "GPU(s) IDs: $CUDA_VISIBLE_DEVICES"
echo "=================================================="
echo ""

# Source bashrc to ensure conda works in non-interactive shells
source ~/.bashrc

# Activate conda environment
conda activate ML || { echo "Failed to activate conda environment"; exit 1; }
echo ""

# Print environment information
echo "================= Environment ===================="
echo "Python version: $(python --version | cut -d' ' -f2)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "CUDA version: $(nvcc --version | grep release | awk '{print $6}' | cut -c2-)"
echo "=================================================="
echo ""

# Get model and dataset arguments
MODEL=$1
DATASET=$2
SPECTRUM=$3

# Run YOLO training
echo "Starting YOLO training on $(date '+%A, %-d %B %Y, %I:%M:%S %p')"

if [ -z "$SPECTRUM" ]; then
    python yolo.py --model $MODEL --dataset $DATASET || { echo "YOLO training failed with exit code $?"; exit 1; }
else
    python yolo.py --model $MODEL --dataset $DATASET --spectrum $SPECTRUM || { echo "YOLO training failed with exit code $?"; exit 1; }
fi

echo "Job completed successfully on $(date '+%A, %-d %B %Y, %I:%M:%S %p')"
echo ""
