import gradio as gr
import os
import torch
from llama_models.llama3.reference_impl.generation import Llama
from llama_models.datatypes import RawMessage

class LlamaChat:
    def __init__(self, model_path="C:\\Users\\lkxv3\\.llama\\checkpoints\\Llama3.2-11B-Vision-Instruct"):
        """
        Initialize the Llama chat model
        :param model_path: Path to the downloaded model
        """
        print("Loading model...")
        # Set environment variables for distributed setup
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        
        self.generator = Llama.build(
            ckpt_dir=model_path,
            max_seq_len=512,
            max_batch_size=4,
            device="cpu"  # Force CPU usage
        )
        print("Model loaded successfully!")

    def generate_response(self, prompt, max_length=200, temperature=0.7):
        """
        Generate a response from the model
        :param prompt: User's input prompt
        :param max_length: Maximum length of the generated response
        :param temperature: Controls randomness in the output
        :return: Generated response
        """
        # Create a dialog with the user's prompt
        dialog = [RawMessage(role="user", content=prompt)]
        
        # Generate response
        result = self.generator.chat_completion(
            dialog,
            max_gen_len=max_length,
            temperature=temperature,
            top_p=0.95,
        )
        
        # Return the generated response
        return result.generation.content

def create_gradio_interface():
    # Initialize the chat model
    chat = LlamaChat()
    
    # Define the chat function
    def chat_with_model(message, history):
        try:
            response = chat.generate_response(message)
            history.append((message, response))
            return "", history
        except Exception as e:
            return "", history + [(message, f"Error: {str(e)}")]
    
    # Create the Gradio interface
    with gr.Blocks(title="Llama Chat Interface", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🤖 Llama Chat Interface")
        gr.Markdown("Chat with Llama 3.2 11B Vision Instruct model")
        
        # Chat interface
        chatbot = gr.Chatbot(
            [],
            elem_id="chatbot",
            bubble_full_width=False,
            avatar_images=(None, "🤖"),
            height=500
        )
        
        # Input components
        with gr.Row():
            txt = gr.Textbox(
                show_label=False,
                placeholder="Type your message here...",
                container=False
            )
            submit_btn = gr.Button("Send", variant="primary")
        
        # Clear button
        clear_btn = gr.Button("Clear Chat")
        
        # Event handlers
        txt.submit(chat_with_model, [txt, chatbot], [txt, chatbot])
        submit_btn.click(chat_with_model, [txt, chatbot], [txt, chatbot])
        clear_btn.click(lambda: None, None, chatbot, queue=False)
        
        # Add some example prompts
        gr.Examples(
            examples=[
                "What is artificial intelligence?",
                "Write a short story about a robot.",
                "Explain quantum computing in simple terms.",
                "What are the benefits of renewable energy?"
            ],
            inputs=txt
        )
    
    return interface

def main():
    # Create and launch the Gradio interface
    interface = create_gradio_interface()
    interface.launch(
        share=True,  # Enable sharing
        server_name="0.0.0.0",  # Allow external access
        server_port=7860  # Specify port
    )

if __name__ == "__main__":
    main() 