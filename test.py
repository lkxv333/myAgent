from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline
import torch
from diffusers import StableDiffusionPipeline
import os

# Load the Llama model and tokenizer
def load_llama_model(model_name="meta-llama/Meta-Llama-3-8B-Instruct", device="cpu"):
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

# Load the Stable Diffusion model
def load_stable_diffusion(model_name="runwayml/stable-diffusion-v1-5", device="cpu"):
    """
    Load the Stable Diffusion model for image generation.
    :param model_name: The name of the Stable Diffusion model from Hugging Face.
    :param device: The device to load the model on ('cpu' or 'cuda').
    :return: The Stable Diffusion pipeline.
    """
    print(f"Loading Stable Diffusion model '{model_name}' on {device}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)
    print("Stable Diffusion model loaded successfully!")
    return pipe

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

# Define a function for image generation
def generate_image(prompt, pipe, output_dir="generated_images"):
    """
    Generate an image using Stable Diffusion.
    :param prompt: The text prompt for image generation.
    :param pipe: The Stable Diffusion pipeline.
    :param output_dir: Directory to save the generated images.
    :return: Path to the generated image.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the image
    image = pipe(prompt).images[0]
    
    # Save the image
    image_path = os.path.join(output_dir, f"{prompt[:50].replace(' ', '_')}.png")
    image.save(image_path)
    return image_path

# Main function to demonstrate usage
if __name__ == "__main__":
    # Specify the model names and device
    llama_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    sd_model_name = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Load Llama model
        llama_model, llama_tokenizer = load_llama_model(llama_model_name, device)
        
        # Load Stable Diffusion model
        sd_pipe = load_stable_diffusion(sd_model_name, device)
        
        # Example purposes
        text_prompts = {
            "text_generation": "Write a short story about a futuristic AI.",
            "summarization": "Summarize the following text: Artificial intelligence is transforming industries...",
            "question_answering": "What is the capital of France?"
        }
        
        # Example image prompts
        image_prompts = [
            "A futuristic AI robot in a cyberpunk city",
            "A beautiful sunset over a digital landscape",
            "An abstract representation of artificial intelligence"
        ]
        
        # Generate text responses
        print("\nGenerating text responses:")
        for purpose, prompt in text_prompts.items():
            print(f"\nPurpose: {purpose}")
            response = generate_text(prompt, llama_model, llama_tokenizer, max_length=200, device=device)
            print(f"Response: {response}")
        
        # Generate images
        print("\nGenerating images:")
        for prompt in image_prompts:
            print(f"\nGenerating image for prompt: {prompt}")
            image_path = generate_image(prompt, sd_pipe)
            print(f"Image saved at: {image_path}")
            
    except Exception as e:
        print(f"Error: {e}")
        exit(1)