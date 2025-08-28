import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- Configuration ---
# The base model used for fine-tuning.
BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B"
# The Hugging Face repository ID of your fine-tuned model.
ADAPTER_MODEL_ID = "likhonsheikh/sheikh-coder"
# Your Hugging Face token, required for gated models.
HF_TOKEN = "YOUR_HUGGING_FACE_TOKEN"

def run_inference(instruction, input_text=""):
    """
    Loads the fine-tuned model and runs inference on a given instruction.
    """
    print("Loading model and tokenizer...")

    # Configure quantization to load the base model in 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the LoRA adapter
    model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_ID)

    print("Model and tokenizer loaded successfully.")
    print("-" * 20)

    # --- Create the prompt ---
    # The prompt must be in the same XML format used for training.
    # Note: We are only providing the start of the <output> tag to prompt the model.
    prompt = f"""<example id="test-001">
  <instruction>{instruction}</instruction>
  <input>{input_text}</input>
  <output>
"""

    print(f"Prompt:\n{prompt}")

    # --- Run Inference ---
    # Note: This assumes you are running on a machine with a CUDA-enabled GPU.
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode and print the result
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("-" * 20)
    print(f"Model Output:\n{result}")


if __name__ == "__main__":
    # --- Example Usage ---
    # NOTE: You will need to replace HF_TOKEN with your actual token.
    # If you get an error, make sure you have accepted the license for Llama 3 on Hugging Face.
    if HF_TOKEN == "YOUR_HUGGING_FACE_TOKEN":
        print("="*50)
        print("WARNING: Please replace 'YOUR_HUGGING_FACE_TOKEN' with your actual token in the script.")
        print("="*50)
    else:
        example_instruction = "Translate the following English text to Bangla."
        example_input = "Hello, how are you?"
        run_inference(example_instruction, example_input)
