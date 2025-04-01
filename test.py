from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline
import torch

# Load the Llama model and tokenizer
def load_llama_model(model_name="meta-llama/Llama-2-7b-hf", device="cpu"):
    """
    Load the Llama model and tokenizer for on-device use.
    :param model_name: The name of the Llama model from Hugging Face.
    :param device: The device to load the model on ('cpu' or 'cuda').
    :return: The model and tokenizer.
    """
    print(f"Loading model '{model_name}' on {device}...")
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model = model.to(device)
    print("Model loaded successfully!")
    return model, tokenizer

# Define a function for text generation
def generate_text(prompt, model, tokenizer, max_length=100, device="cpu"):
    """
    Generate text using the Llama model.
    :param prompt: The input prompt for text generation.
    :param model: The Llama model.
    :param tokenizer: The tokenizer for the model.
    :param max_length: The maximum length of the generated text.
    :param device: The device to run the model on.
    :return: The generated text.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs["input_ids"], max_length=max_length, num_return_sequences=1, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Main function to demonstrate usage
if __name__ == "__main__":
    # Specify the model name and device
    model_name = "meta-llama/Llama-3.1-8B"  # Replace with the desired Llama model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    except OSError as e:
        print(f"Error loading model or tokenizer: {e}")
        exit(1)

    # Load the model and tokenizer
    model, tokenizer = load_llama_model(model_name, device)

    # Example purposes
    purposes = {
        "text_generation": "Write a short story about a futuristic AI.",
        "summarization": "Summarize the following text: Artificial intelligence is transforming industries...",
        "question_answering": "What is the capital of France?"
    }

    # Iterate through purposes and generate responses
    for purpose, prompt in purposes.items():
        print(f"\nPurpose: {purpose}")
        response = generate_text(prompt, model, tokenizer, max_length=200, device=device)
        print(f"Response: {response}")