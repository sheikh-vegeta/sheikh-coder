# Project Architecture

This document provides an overview of the architecture of the `sheikh-coder` training pipeline.

## Core Components

The pipeline consists of three main Python scripts:

1.  **`train_sheikh.py`**: This is the main entry point for the application. It is a command-line interface (CLI) that parses user arguments (like model name, dataset, etc.) and orchestrates the training process by calling the `trainer` module. It is also responsible for handling the Hugging Face authentication token, prioritizing the command-line argument over environment variables.

2.  **`trainer.py`**: This script contains the core logic for the training process. Its `run` function executes the following steps:
    *   Logs into the Hugging Face Hub.
    *   Loads a base model and tokenizer, applying 4-bit quantization (QLoRA) for memory efficiency.
    *   Loads a dataset from the Hugging Face Hub.
    *   Preprocesses the dataset by formatting each sample into a structured XML string.
    *   Configures the LoRA parameters using PEFT (`peft`).
    *   Sets up and runs the training using the `SFTTrainer` from the TRL library (`trl`).
    *   Saves the final trained model adapter. If configured, it also pushes the model to the Hugging Face Hub.

3.  **`inference.py`**: This script provides an example of how to use a fine-tuned model for inference. It loads the base model, applies the trained LoRA adapter from the Hub, and generates text from a sample prompt.

## Data Flow

The typical data flow is as follows:

1.  A user or an automated workflow (like GitHub Actions) executes `train_sheikh.py` with specific arguments.
2.  `train_sheikh.py` calls `trainer.run()`.
3.  The `trainer` module pulls the base model and dataset from the Hugging Face Hub.
4.  The model is fine-tuned on the preprocessed data.
5.  The resulting model adapter (the "trained model") is saved to a local directory and/or pushed back to a specified repository on the Hugging Face Hub.
