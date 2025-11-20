"""
Image and text processing utilities for PaliGemma.

This module handles:
- Image preprocessing (resizing, normalization)
- Text tokenization with special image tokens
- Combining visual and textual inputs for the model
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Union, Tuple, Iterable


# Standard normalization constants for ImageNet-pretrained models
# These values center the data around 0 with standard deviation 1
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]  # Mean for R, G, B channels
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]   # Std dev for R, G, B channels


def add_image_tokens_to_prompt(
    prefix_prompt: str, bos_token: str, image_seq_len: int, image_token: str
) -> str:
    """
    Prepend image tokens to the text prompt.
    
    PaliGemma processes images as a sequence of special tokens before the text.
    For a 224x224 image with 16x16 patches, we get 196 image tokens.
    
    Format: <image><image>...<image><bos>actual prompt text\n
    
    Args:
        prefix_prompt: The actual text prompt from the user
        bos_token: Beginning of sequence token (e.g., "<bos>")
        image_seq_len: Number of image patch tokens (e.g., 196)
        image_token: Special token representing image patches (e.g., "<image>")
    
    Returns:
        Complete prompt string with image tokens prepended
    
    Example:
        Input: prefix_prompt="describe this image", image_seq_len=3
        Output: "<image><image><image><bos>describe this image\n"
    """
    # Repeat image token image_seq_len times, then add BOS, prompt, and newline
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


def rescale(
    image: np.ndarray, scale: float, dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Rescale pixel values by a constant factor.
    
    Typically used to normalize uint8 pixel values (0-255) to float range (0-1).
    
    Args:
        image: Input image array, typically with values in [0, 255]
        scale: Scaling factor (commonly 1/255.0 = 0.00392...)
        dtype: Target data type for output (default: float32)
    
    Returns:
        Rescaled image array with values in [0, 1] if scale=1/255
    
    Example:
        image with values [0, 128, 255] and scale=1/255
        becomes [0.0, 0.502, 1.0]
    """
    # Multiply all pixel values by scale factor
    rescaled_image = image * scale
    # Cast to desired dtype (important for memory and precision)
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image


def resize(
    image: Image.Image,
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    reducing_gap: int = None,
) -> Image.Image:
    """
    Resize PIL Image to specified dimensions.
    
    Args:
        image: Input PIL Image object
        size: Target size as (height, width) tuple
        resample: Resampling filter (e.g., BICUBIC for quality)
        reducing_gap: Optimization for downscaling (None for default)
    
    Returns:
        Resized PIL Image object
    
    Note:
        PIL.Image.resize expects (width, height) but we receive (height, width),
        so we swap the order when calling resize.
    """
    height, width = size
    # PIL resize takes (width, height) order, so we swap
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_image


def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:
    """
    Normalize image using channel-wise mean and standard deviation.
    
    Applies the transformation: output = (input - mean) / std
    This centers the data around 0 and scales it to unit variance.
    
    Args:
        image: Input image array [Height, Width, Channels]
        mean: Mean value(s) for normalization, one per channel or single value
        std: Standard deviation(s) for normalization, one per channel or single value
    
    Returns:
        Normalized image array with approximately zero mean and unit variance
    
    Example:
        For ImageNet: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        Input pixel value 0.5 becomes (0.5 - 0.5) / 0.5 = 0.0
        Input pixel value 1.0 becomes (1.0 - 0.5) / 0.5 = 1.0
    """
    # Convert mean and std to numpy arrays with same dtype as image
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    
    # Apply normalization formula
    # Broadcasting handles per-channel mean/std automatically
    image = (image - mean) / std
    return image


def process_images(
    images: List[Image.Image],
    size: Dict[str, int] = None,
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: Union[float, List[float]] = None,
    image_std: Union[float, List[float]] = None,
) -> List[np.ndarray]:
    """
    Complete image preprocessing pipeline for model input.
    
    Pipeline steps:
        1. Resize images to fixed dimensions
        2. Convert to numpy arrays
        3. Rescale pixel values to [0, 1]
        4. Normalize using mean and std
        5. Transpose to channel-first format [C, H, W]
    
    Args:
        images: List of PIL Image objects
        size: Target size as [height, width]
        resample: Resampling method (e.g., BICUBIC)
        rescale_factor: Factor to rescale pixels (typically 1/255)
        image_mean: Mean values for normalization
        image_std: Std deviation values for normalization
    
    Returns:
        List of preprocessed numpy arrays, each with shape [Channels, Height, Width]
    """
    # Step 1: Resize all images to same dimensions
    height, width = size[0], size[1]
    images = [
        resize(image=image, size=(height, width), resample=resample) 
        for image in images
    ]
    
    # Step 2: Convert PIL Images to numpy arrays
    # Results in shape [Height, Width, Channels] with uint8 values
    images = [np.array(image) for image in images]
    
    # Step 3: Rescale pixel values from [0, 255] to [0, 1]
    # This makes the data more suitable for neural network processing
    images = [rescale(image, scale=rescale_factor) for image in images]
    
    # Step 4: Normalize images to have mean 0 and standard deviation 1
    # This helps with training stability and convergence
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    
    # Step 5: Transpose from [H, W, C] to [C, H, W]
    # PyTorch expects channel-first format: [Channels, Height, Width]
    # This is different from numpy/PIL default of [Height, Width, Channels]
    images = [image.transpose(2, 0, 1) for image in images]
    
    return images


class PaliGemmaProcessor:
    """
    Unified processor for PaliGemma that handles both images and text.
    
    Responsibilities:
        - Tokenize text prompts
        - Preprocess images into patches
        - Add special image tokens to prompts
        - Combine visual and textual inputs
    
    Attributes:
        IMAGE_TOKEN: Special token string representing image patches
        image_seq_length: Number of image tokens per image
        image_size: Expected input image size (height and width)
        tokenizer: Text tokenizer for processing prompts
        image_token_id: Numerical ID of the image token
    """

    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        """
        Initialize the processor with tokenizer and image configuration.
        
        Args:
            tokenizer: HuggingFace tokenizer instance
            num_image_tokens: Number of patch tokens per image (e.g., 196 for 14x14 patches)
            image_size: Image dimension for resizing (assumed square)
        """
        super().__init__()

        # Store image tokenization parameters
        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        # Add special image token to tokenizer vocabulary
        # This allows the model to recognize where image embeddings should go
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        
        # Add location tokens for object detection (bounding box coordinates)
        # Format: <loc0000> to <loc1023> for 1024 possible coordinate values
        EXTRA_TOKENS = [f"<loc{i:04d}>" for i in range(1024)]
        
        # Add segmentation tokens for pixel-level segmentation masks
        # Format: <seg000> to <seg127> for 128 possible segment IDs
        EXTRA_TOKENS += [f"<seg{i:03d}>" for i in range(128)]
        
        # Register all extra tokens with tokenizer
        tokenizer.add_tokens(EXTRA_TOKENS)
        
        # Get numerical ID for image token (needed for detecting image positions)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        
        # Disable automatic BOS/EOS addition (we'll add them manually)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ) -> dict:
        """
        Process images and text into model-ready inputs.
        
        This method:
            1. Preprocesses images into normalized tensors
            2. Adds image tokens to text prompts
            3. Tokenizes the combined text
            4. Returns dictionary with pixel_values, input_ids, and attention_mask
        
        Args:
            text: List of text prompts (currently expects single prompt)
            images: List of PIL Images (currently expects single image)
            padding: Padding strategy ("longest", "max_length", etc.)
            truncation: Whether to truncate sequences exceeding max length
        
        Returns:
            Dictionary containing:
                - pixel_values: Image tensor [Batch, Channels, Height, Width]
                - input_ids: Token IDs [Batch, Sequence_Length]
                - attention_mask: Attention mask [Batch, Sequence_Length]
        
        Raises:
            AssertionError: If number of images doesn't match number of prompts
        """
        # Currently only supports single image-text pairs
        assert len(images) == 1 and len(text) == 1, (
            f"Received {len(images)} images for {len(text)} prompts."
        )

        # Process images through complete preprocessing pipeline
        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),  # Resize to model's expected size
            resample=Image.Resampling.BICUBIC,         # High-quality resampling
            rescale_factor=1 / 255.0,                  # Scale to [0, 1]
            image_mean=IMAGENET_STANDARD_MEAN,         # Normalize with ImageNet stats
            image_std=IMAGENET_STANDARD_STD,
        )
        
        # Stack list of arrays into single batch
        # Shape: [Batch_Size=1, Channels=3, Height=224, Width=224]
        pixel_values = np.stack(pixel_values, axis=0)
        
        # Convert numpy array to PyTorch tensor
        pixel_values = torch.tensor(pixel_values)

        # Prepend image tokens to each text prompt
        # Result: "<image><image>...<bos>prompt text\n"
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # Tokenize the prepared strings
        # Returns PyTorch tensors for input_ids and attention_mask
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",  # Return PyTorch tensors
            padding=padding,       # Pad to longest sequence in batch
            truncation=truncation, # Truncate if exceeds max length
        )

        # Combine visual and textual inputs into single dictionary
        return_data = {"pixel_values": pixel_values, **inputs}

        return return_data