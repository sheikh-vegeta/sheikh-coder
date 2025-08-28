import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
from huggingface_hub import login

def format_instruction_xml(instruction: str, input_text: str = "", output_text: str = "", example_id: str = None) -> str:
    """
    Formats an instruction set into an XML-like string for Sheikh model training.
    """
    example_id_attr = f' id="{example_id}"' if example_id else ""
    xml = f"<example{example_id_attr}>\n"
    xml += f"  <instruction>{instruction}</instruction>\n"
    if input_text:
        xml += f"  <input>{input_text}</input>\n"
    if output_text:
        xml += f"  <output>{output_text}</output>\n"
    xml += "</example>"
    return xml

def run(args):
    """
    The main training function.
    """
    print("Starting training process...")

    # 1. Hugging Face Login
    print(f"Logging into Hugging Face Hub with token.")
    login(token=args.hf_token)

    # 2. Load Model and Tokenizer
    print(f"Loading base model: {args.model_name}")

    # Configure quantization for QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Load and Preprocess Dataset
    print(f"Loading dataset: {args.dataset_name}")
    # Using validation split as it is smaller and available for gaia-benchmark/GAIA
    dataset = load_dataset(args.dataset_name, split="validation")

    # For demonstration, we'll just use a small subset of the data
    dataset = dataset.select(range(100))

    def format_gaia_sample(sample, index):
        instruction = "You are a helpful assistant. Your task is to answer the following question."
        input_text = sample.get("Question", "")
        output_text = sample.get("Final Rationale", "")
        example_id = f"{index:03d}"

        return {"text": format_instruction_xml(
            instruction=instruction,
            input_text=input_text,
            output_text=output_text,
            example_id=example_id
        )}

    # We need to manually create the 'text' field for the SFTTrainer, now in XML format.
    dataset = dataset.map(format_gaia_sample, with_indices=True)

    # 4. Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. Set up Training
    print("Setting up training arguments...")
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=10,
        num_train_epochs=1,
        fp16=True,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    # 6. Run Training
    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning finished.")

    # 7. Save Artifacts
    print(f"Saving model and tokenizer to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Training process finished successfully.")
