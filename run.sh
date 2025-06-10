#!/bin/bash

DEFAULT_DATA_DIR="elope_dataset"
DEFAULT_N_VALID=5
ZIP_FILE="elope_dataset.zip"
ZENODO_URL="https://zenodo.org/record/15421707/files/elope_dataset.zip?download=1"
PYTHON_SCRIPT="src/train.py"
OUTPUT_DIR="outputs"

function download_dataset() {
    local DATA_DIR="${1:-$DEFAULT_DATA_DIR}"
    local N_VALID="${2:-$DEFAULT_N_VALID}"
    local VENV_DIR=".venv"

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

    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment in $VENV_DIR..."
        python3 -m venv "$VENV_DIR"
    else
        echo "Virtual environment already exists."
    fi

    source "$VENV_DIR/bin/activate"
    echo "Installing requirements from requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
    deactivate

    echo "Environment setup complete."
}

function train() {
    echo "Starting training..."
    python "$PYTHON_SCRIPT" --mode train --folder_path "$DEFAULT_DATA_DIR/train" --output_dir "$OUTPUT_DIR/train" --epochs 20 --batch_size 16 --use_range
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
    if [ -z "$1" ]; then
        echo "Error: Please provide checkpoint path for testing"
        exit 1
    fi
    echo "Starting test with checkpoint $1 ..."
    python "$PYTHON_SCRIPT" --mode test --folder_path "$DEFAULT_DATA_DIR/test" --checkpoint_path "$1" --batch_size 16 --use_range --output_dir "$OUTPUT_DIR/test"
}

case "$1" in
    download)
        download_dataset "$2" "$3"
        ;;
    train)
        train
        ;;
    validate)
        validate "$2"
        ;;
    test)
        test "$2"
        ;;
    *)
        echo "Usage: $0 {download [data_dir] [n_valid]|train|validate <checkpoint>|test <checkpoint>}"
        exit 1
esac
