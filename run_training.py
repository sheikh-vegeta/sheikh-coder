import argparse
import trainer

def main():
    """
    Main function to parse arguments and trigger the training process.
    """
    parser = argparse.ArgumentParser(description="Fine-tune a model using LoRA.")

    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="The name of the base model to use from Hugging Face."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="gaia-benchmark/GAIA",
        help="The name of the dataset to use for fine-tuning."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        required=True,
        help="The Hugging Face API token for accessing gated models and datasets."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="The directory to save the output artifacts (model, checkpoints, logs)."
    )

    args = parser.parse_args()

    print("Training script started.")
    trainer.run(args)
    print("Training script finished.")

if __name__ == "__main__":
    main()
