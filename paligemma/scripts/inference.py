"""
Model loading and inference utilities for PaliGemma.

This module provides functions to:
- Load pretrained models from HuggingFace format
- Prepare inputs for inference
- Run inference with various sampling strategies
"""

import torch
from PIL import Image
import json
import glob
import os
from safetensors import safe_open
from typing import Tuple
from transformers import AutoTokenizer

from config.config import PaliGemmaConfig
from models.modeling import PaliGemmaForConditionalGeneration
from helper.processing import PaliGemmaProcessor
from helper.utils import KVCache, sample_top_p


def load_hf_model(
    model_path: str, device: str
) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    """
    Load PaliGemma model from HuggingFace format.
    
    This function:
        1. Loads the tokenizer
        2. Finds all safetensors weight files
        3. Loads weights into a dictionary
        4. Creates model from config
        5. Loads state dict into model
        6. Ties embedding weights
    
    Args:
        model_path: Path to directory containing model files
                   Should contain: config.json, *.safetensors, tokenizer files
        device: Device to load model on ("cpu", "cuda", or "mps")
    
    Returns:
        Tuple of (model, tokenizer)
        - model: PaliGemmaForConditionalGeneration instance
        - tokenizer: HuggingFace tokenizer
    """
    # Step 1: Load tokenizer with right padding
    # Right padding is important for batch processing
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right", "Tokenizer must use right padding"

    # Step 2: Find all safetensors files in model directory
    # Safetensors is a safe, fast format for storing tensors
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # Step 3: Load all tensors from safetensors files
    # Collect all weights into a single dictionary
    tensors = {}
    for safetensors_file in safetensors_files:
        # Open safetensors file
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            # Iterate over all tensors in the file
            for key in f.keys():
                # Load each tensor and store with its key
                tensors[key] = f.get_tensor(key)

    # Step 4: Load model configuration from JSON
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        # Create PaliGemmaConfig from loaded dictionary
        config = PaliGemmaConfig(**model_config_file)

    # Step 5: Create model architecture from config and move to device
    model = PaliGemmaForConditionalGeneration(config).to(device)

    # Step 6: Load pretrained weights into model
    # strict=False allows for missing/unexpected keys (e.g., tied weights)
    model.load_state_dict(tensors, strict=False)

    # Step 7: Tie embedding and language modeling head weights
    # This shares parameters between input embeddings and output projection
    model.tie_weights()

    return (model, tokenizer)


def move_inputs_to_device(model_inputs: dict, device: str) -> dict:
    """
    Move all input tensors to specified device.
    
    Args:
        model_inputs: Dictionary of input tensors
        device: Target device ("cpu", "cuda", or "mps")
    
    Returns:
        Dictionary with all tensors moved to device
    """
    # Iterate through all items and move tensors to device
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs


def get_model_inputs(
    processor: PaliGemmaProcessor, 
    prompt: str, 
    image_file_path: str, 
    device: str
) -> dict:
    """
    Prepare inputs for model inference.
    
    Args:
        processor: PaliGemmaProcessor instance
        prompt: Text prompt describing the task
        image_file_path: Path to input image file
        device: Device to place tensors on
    
    Returns:
        Dictionary containing:
            - input_ids: Token IDs [1, Seq_Len]
            - attention_mask: Attention mask [1, Seq_Len]
            - pixel_values: Processed image [1, 3, Size, Size]
    """
    # Load image from file
    image = Image.open(image_file_path)
    
    # Wrap in lists (processor expects batch format)
    images = [image]
    prompts = [prompt]
    
    # Process through PaliGemma processor
    # This adds image tokens, tokenizes text, and processes image
    model_inputs = processor(text=prompts, images=images)
    
    # Move all tensors to target device
    model_inputs = move_inputs_to_device(model_inputs, device)
    
    return model_inputs


def test_inference(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    """
    Run inference to generate text from image and prompt.
    
    This function implements autoregressive generation:
        1. Process image and prompt (prefill phase)
        2. Generate tokens one at a time
        3. Stop at EOS token or max length
        4. Decode and print result
    
    Args:
        model: PaliGemma model instance
        processor: Processor for inputs
        device: Device to run on
        prompt: Text prompt for the task
        image_file_path: Path to input image
        max_tokens_to_generate: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
                    Only used if do_sample=True
        top_p: Nucleus sampling threshold (0.0 to 1.0)
              Only used if do_sample=True
        do_sample: If True, use sampling; if False, use greedy decoding
    
    Process:
        - Greedy decoding (do_sample=False): Always pick most likely token
        - Sampling (do_sample=True): Sample from distribution with temperature/top_p
    """
    # Prepare model inputs
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    
    # Extract components
    input_ids = model_inputs["input_ids"]          # [1, Seq_Len]
    attention_mask = model_inputs["attention_mask"]  # [1, Seq_Len]
    pixel_values = model_inputs["pixel_values"]      # [1, 3, H, W]

    # Initialize KV cache for efficient generation
    kv_cache = KVCache()

    # Stop token ID (marks end of generation)
    stop_token = processor.tokenizer.eos_token_id
    
    # List to collect generated token IDs
    generated_tokens = []

    # Autoregressive generation loop
    for _ in range(max_tokens_to_generate):
        # Forward pass through model
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        
        # Update cache with new key-value pairs
        kv_cache = outputs["kv_cache"]
        
        # Get logits for next token (last position only)
        # [Batch_Size, Seq_Len, Vocab_Size] -> [Batch_Size, Vocab_Size]
        next_token_logits = outputs["logits"][:, -1, :]
        
        # Sample next token based on strategy
        if do_sample:
            # Sampling mode: use temperature and top-p
            
            # Apply temperature scaling
            # Higher temperature = flatter distribution (more random)
            # Lower temperature = sharper distribution (more deterministic)
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            
            # Apply top-p (nucleus) sampling
            # Sample from smallest set of tokens with cumulative prob > p
            next_token = sample_top_p(next_token_logits, top_p)
        else:
            # Greedy decoding: always pick most likely token
            # [Batch_Size, Vocab_Size] -> [Batch_Size, 1]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Validate shape
        assert next_token.size() == (1, 1), f"Expected shape (1, 1), got {next_token.size()}"
        
        # Remove batch dimension: [1, 1] -> [1]
        next_token = next_token.squeeze(0)
        
        # Add to generated sequence
        generated_tokens.append(next_token)
        
        # Check for stop condition
        if next_token.item() == stop_token:
            break
        
        # Prepare input for next iteration
        # Use the generated token as input
        # [1] -> [1, 1]
        input_ids = next_token.unsqueeze(-1)
        
        # Extend attention mask (attend to new token)
        # Concatenate [1, Old_Len] with [1, 1] -> [1, Old_Len + 1]
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
        )

    # Concatenate all generated tokens
    # List of [1] tensors -> [Num_Generated_Tokens]
    generated_tokens = torch.cat(generated_tokens, dim=-1)
    
    # Decode tokens to text
    # skip_special_tokens removes <eos>, <pad>, etc.
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Print prompt and generated text
    print(prompt + "  " + decoded)