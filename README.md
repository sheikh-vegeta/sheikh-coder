# Sheikh Coder

`sheikh-coder` is a project focused on fine-tuning large language models with a specialization in culturally-aware knowledge, particularly for the Bangla language and context. It provides a complete pipeline for fine-tuning, from data preparation to model training and inference.

## Project Overview

This repository contains scripts to:
- **Fine-tune** large language models like Llama 3 or Gemma using LoRA/QLoRA for memory-efficient training.
- **Format** training data using a clear, semantic XML-style structure.
- **Run inference** with the fine-tuned models.
- **Push** trained models directly to the Hugging Face Hub.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd sheikh-coder
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Usage

### Training

The `run_training.py` script is the main entry point for starting a fine-tuning job.

**Example Command:**
```bash
python run_training.py \
    --model_name "meta-llama/Meta-Llama-3-8B" \
    --dataset_name "gaia-benchmark/GAIA" \
    --hf_token "YOUR_HUGGING_FACE_TOKEN" \
    --output_dir "./sheikh-coder-finetuned" \
    --push_to_hub \
    --hub_model_id "likhonsheikh/sheikh-coder"
```

**Arguments:**
- `--model_name`: The base model to fine-tune (e.g., `meta-llama/Meta-Llama-3-8B`).
- `--dataset_name`: The dataset to use for training (e.g., `gaia-benchmark/GAIA`).
- `--hf_token`: (Required) Your Hugging Face API token for accessing gated models.
- `--output_dir`: The local directory where the trained model adapter will be saved.
- `--push_to_hub`: If included, the script will push the final model to the Hugging Face Hub.
- `--hub_model_id`: The repository ID on the Hub to push the model to.

### Inference

The `inference.py` script demonstrates how to load a fine-tuned model from the Hub and run inference.

**Before running:**
1.  Open the `inference.py` script.
2.  Replace the `HF_TOKEN` placeholder with your actual Hugging Face token.
3.  Ensure the `ADAPTER_MODEL_ID` points to the correct repository of your fine-tuned model.

**Run the script:**
```bash
python inference.py
```
The script will load the model and run a sample inference with a pre-defined prompt, printing the output to the console.
