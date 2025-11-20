"""
Main entry point for PaliGemma inference.

This script:
1. Authenticates with HuggingFace (for downloading models)
2. Sets up inference parameters
3. Loads the model and processor
4. Runs inference on an image with a prompt
"""

import torch
from huggingface_hub import notebook_login

from scripts.inference import load_hf_model, test_inference
from helper.processing import PaliGemmaProcessor


def main():
    """
    Main function to run PaliGemma inference.
    
    Configuration:
        - model_path: Path to pretrained model directory
        - prompt: Text prompt describing the vision task
        - image_file_path: Path to input image
        - max_tokens_to_generate: Maximum length of generated text
        - temperature: Sampling temperature (higher = more random)
        - top_p: Nucleus sampling threshold
        - do_sample: Whether to use sampling (True) or greedy decoding (False)
        - only_cpu: Force CPU usage even if GPU is available
    """
    
    # Step 1: Authenticate with HuggingFace Hub
    # Required for downloading gated models
    # This will prompt for login if not already authenticated
    notebook_login()

    # Step 2: Configure inference parameters
    
    # Path to model directory containing:
    # - config.json
    # - model weights (*.safetensors)
    # - tokenizer files
    model_path = "./paligemma-3b-pt-224"
    
    # Task prompt - this describes what the model should do
    # Examples:
    #   "describe this image"
    #   "what color is the cat?"
    #   "detect person"
    prompt = "The number of kittens in the image is"
    
    # Path to input image file
    image_file_path = "./kitten.jpg"
    
    # Generation parameters
    max_tokens_to_generate = 500  # Maximum output length
    temperature = 0.1              # Low temp = more deterministic
    top_p = 0.7                    # Nucleus sampling threshold
    do_sample = False              # Use greedy decoding
    only_cpu = False               # Allow GPU usage
    
    # Step 3: Determine device to use
    device = "cpu"  # Default to CPU
    
    if not only_cpu:
        # Check for available accelerators
        if torch.cuda.is_available():
            device = "cuda"  # NVIDIA GPU
        elif torch.backends.mps.is_available():
            device = "mps"   # Apple Silicon GPU
    
    print("Device in use:", device)

    # Step 4: Load model and tokenizer
    print(f"Loading model from {model_path}")
    model, tokenizer = load_hf_model(model_path, device)
    
    # Move model to device and set to evaluation mode
    model = model.to(device).eval()

    # Step 5: Create processor
    # Processor handles both image preprocessing and text tokenization
    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    # Step 6: Run inference
    print("Running inference")
    with torch.no_grad():  # Disable gradient computation for inference
        test_inference(
            model=model,
            processor=processor,
            device=device,
            prompt=prompt,
            image_file_path=image_file_path,
            max_tokens_to_generate=max_tokens_to_generate,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )


if __name__ == "__main__":
    """
    Entry point when script is run directly.
    
    Usage:
        python main.py
    
    Make sure to:
        1. Install required packages (torch, transformers, PIL, safetensors)
        2. Download model to ./paligemma-3b-pt-224/
        3. Prepare input image at ./kitten.jpg
        4. Authenticate with HuggingFace (notebook_login will prompt)
    """
    main()