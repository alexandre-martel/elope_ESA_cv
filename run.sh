#!/bin/bash

DEFAULT_DATA_DIR="elope_dataset"
DEFAULT_N_VALID=5
ZIP_FILE="elope_dataset.zip"
ZENODO_URL="https://zenodo.org/record/15421707/files/elope_dataset.zip?download=1"
PYTHON_SCRIPT="src/train.py"
OUTPUT_DIR="outputs"
CONDA_ENV_NAME="elope_env"
PYTHON_VERSION="3.10" 

function setup_conda_env() {
    if conda info --envs | grep -q "$CONDA_ENV_NAME"; then
        echo "Conda environment '$CONDA_ENV_NAME' already exists."
    else
        echo "Creating conda environment '$CONDA_ENV_NAME' with Python $PYTHON_VERSION..."
        conda create -y -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION"
    fi

    echo "Activating conda environment '$CONDA_ENV_NAME'..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"

    echo "Installing requirements from requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt

    echo "Environment setup complete."
}


function download_dataset() {
    local DATA_DIR="${1:-$DEFAULT_DATA_DIR}"
    local N_VALID="${2:-$DEFAULT_N_VALID}"

    mkdir -p "$DATA_DIR"
    mkdir -p "$DATA_DIR/validation"

    echo "Downloading dataset from Zenodo..."
    wget -O "$ZIP_FILE" "$ZENODO_URL"

    echo "Extracting dataset..."
    unzip -o "$ZIP_FILE" -d "$DATA_DIR"

    echo "Cleaning up..."
    rm "$ZIP_FILE"

    echo "Dataset extracted to $DATA_DIR/train"

    local TRAIN_DIR="$DATA_DIR/train"
    local VALID_DIR="$DATA_DIR/validation"

    echo "Moving last $N_VALID files from train to validation..."
    ls "$TRAIN_DIR"/*.npz | sort | tail -n "$N_VALID" | while read file; do
        mv "$file" "$VALID_DIR/"
    done

    echo "Validation set created in $VALID_DIR"
}

function train() {
    local EPOCHS="${1:-20}"
    echo "Starting training for $EPOCHS epochs..."
    python "$PYTHON_SCRIPT" --mode train --folder_path "$DEFAULT_DATA_DIR/train" --output_dir "$OUTPUT_DIR/train" --epochs "$EPOCHS" --batch_size 16 --use_range
}

function resume() {
    if [ -z "$1" ]; then
        echo "Error: Please provide checkpoint path to resume training"
        exit 1
    fi
    local CHECKPOINT="$1"
    local EPOCHS="${2:-20}"
    echo "Resuming training from checkpoint $CHECKPOINT for $EPOCHS epochs..."
    python "$PYTHON_SCRIPT" --mode train --folder_path "$DEFAULT_DATA_DIR/train" --output_dir "$OUTPUT_DIR/train" --epochs "$EPOCHS" --batch_size 16 --use_range --resume_from_checkpoint "$CHECKPOINT"
}

function validate() {
    if [ -z "$1" ]; then
        echo "Error: Please provide checkpoint path for validation"
        exit 1
    fi
    echo "Starting validation with checkpoint $1 ..."
    python "$PYTHON_SCRIPT" --mode validate --folder_path "$DEFAULT_DATA_DIR/validation" --checkpoint_path "$1" --batch_size 16 --use_range --output_dir "$OUTPUT_DIR/validate"
}

function test() {
    local CHECKPOINT_PATH="$1"

    if [ -z "$CHECKPOINT_PATH" ]; then
        echo "[INFO] No checkpoint path provided — using latest checkpoint in outputs/train/checkpoints/..."
        CHECKPOINT_PATH=$(ls -t "$OUTPUT_DIR/train/checkpoints"/epoch-*.ckpt 2>/dev/null | head -n 1)

        if [ -z "$CHECKPOINT_PATH" ]; then
            echo "Error: No checkpoint found in outputs/train/checkpoints/"
            exit 1
        fi
    fi

    echo "Starting test with checkpoint: $CHECKPOINT_PATH"
    python "$PYTHON_SCRIPT" \
        --mode test \
        --folder_path "$DEFAULT_DATA_DIR/test" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --batch_size 16 \
        --use_range \
        --output_dir "$OUTPUT_DIR/test"
}





case "$1" in
    setup)
        setup_conda_env
        ;;
    download)
        download_dataset "$2" "$3"
        ;;
    train)
        train "$2"
        ;;
    resume)
        resume "$2" "$3"
        ;;
    validate)
        validate "$2"
        ;;
    test)
        test "$2"
        ;;
    *)
        echo "Usage: $0 {setup|download [data_dir] [n_valid]|train [epochs]|resume <checkpoint> [epochs]|validate <checkpoint>|test <checkpoint>}"
        exit 1
esac