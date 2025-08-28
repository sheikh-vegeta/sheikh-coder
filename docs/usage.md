# Usage Guide

This guide provides detailed instructions on how to use the scripts in this repository.

## Setup

Before running any scripts, ensure you have set up your environment correctly as described in the main `README.md` file (cloning the repo, creating a virtual environment, and installing dependencies).

## Training

The `train_sheikh.py` script is used to fine-tune a model.

### Command-Line Arguments

The script accepts the following arguments:

| Argument | Description | Default |
| --- | --- | --- |
| `--model_name` | The repository ID of the base model on the Hugging Face Hub. | `meta-llama/Meta-Llama-3-8B` |
| `--dataset_name`| The repository ID of the dataset on the Hugging Face Hub. | `gaia-benchmark/GAIA` |
| `--hf_token` | Your Hugging Face token. If not provided, it will be read from the `HF_TOKEN` environment variable. | `None` |
| `--output_dir` | The local directory where the final model adapter will be saved. | `./sheikh-coder-finetuned` |
| `--push_to_hub` | A boolean flag. If present, the model will be pushed to the Hub. | `False` |
| `--hub_model_id`| The repository ID to push the model to on the Hub. | `likhonsheikh/sheikh-coder` |

### Example

```bash
python train_sheikh.py \
    --model_name "google/gemma-2b" \
    --dataset_name "likhonsheikh/my-custom-dataset" \
    --output_dir "./gemma-finetuned" \
    --push_to_hub
```
*(Note: You must have your `HF_TOKEN` set as an environment variable for the command above to work.)*

## Inference

The `inference.py` script is used to run predictions with a fine-tuned model.

### Configuration

Before running the script, you must configure the following variables inside `inference.py`:

-   `BASE_MODEL_ID`: This should be the same base model that was used for fine-tuning.
-   `ADAPTER_MODEL_ID`: This is the repository ID of your fine-tuned adapter on the Hugging Face Hub.
-   `HF_TOKEN`: Your Hugging Face token.

### Running

Once configured, you can run the script directly:
```bash
python inference.py
```
The script will load the specified model and adapter, and then run a sample inference based on the `example_instruction` and `example_input` variables defined in the script's `if __name__ == "__main__":` block.
