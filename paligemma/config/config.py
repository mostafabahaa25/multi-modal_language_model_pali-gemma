"""
Configuration classes for PaliGemma model components.

This module defines configuration dataclasses for:
- Vision encoder (SigLIP)
- Language model (Gemma)
- Combined multimodal model (PaliGemma)
"""


class SiglipVisionConfig:
    """
    Configuration class for SigLIP Vision Transformer.
    
    This configuration defines the architecture of the vision encoder that processes
    images into patch embeddings.
    
    Args:
        hidden_size: Dimensionality of the encoder layers (default: 768)
        intermediate_size: Dimensionality of the feedforward layer (default: 3072)
        num_hidden_layers: Number of transformer encoder layers (default: 12)
        num_attention_heads: Number of attention heads per layer (default: 12)
        num_channels: Number of input image channels, RGB=3 (default: 3)
        image_size: Expected input image resolution (default: 224)
        patch_size: Size of image patches for tokenization (default: 16)
        layer_norm_eps: Epsilon value for layer normalization (default: 1e-6)
        attention_dropout: Dropout probability for attention weights (default: 0.0)
        num_image_tokens: Total number of image tokens after patching (computed)
    """

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs
    ):
        super().__init__()

        # Core architectural dimensions
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        
        # Image processing parameters
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        
        # Training stability parameters
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        
        # Number of patches the image will be divided into
        self.num_image_tokens = num_image_tokens


class GemmaConfig:
    """
    Configuration class for Gemma language model.
    
    Gemma is a decoder-only transformer model that generates text tokens.
    This configuration supports grouped-query attention for efficiency.
    
    Args:
        vocab_size: Size of the vocabulary
        hidden_size: Dimension of the hidden states
        intermediate_size: Dimension of the MLP feedforward layer
        num_hidden_layers: Number of decoder layers
        num_attention_heads: Number of query attention heads
        num_key_value_heads: Number of key/value heads (for GQA)
        head_dim: Dimension of each attention head (default: 256)
        max_position_embeddings: Maximum sequence length (default: 8192)
        rms_norm_eps: Epsilon for RMS normalization (default: 1e-6)
        rope_theta: Base frequency for rotary embeddings (default: 10000.0)
        attention_bias: Whether to use bias in attention projections (default: False)
        attention_dropout: Dropout rate for attention (default: 0.0)
        pad_token_id: ID for padding tokens (default: None)
    """

    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        
        # Vocabulary and sequence configuration
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        
        # Model dimension configuration
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        
        # Attention configuration (supports Grouped Query Attention)
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads  # GQA: fewer KV heads than Q heads
        
        # Normalization and stability
        self.rms_norm_eps = rms_norm_eps
        
        # Rotary positional encoding configuration
        self.rope_theta = rope_theta
        
        # Attention layer configuration
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        
        # Special tokens
        self.pad_token_id = pad_token_id


class PaliGemmaConfig:
    """
    Configuration class for the complete PaliGemma multimodal model.
    
    PaliGemma combines a vision encoder (SigLIP) with a language model (Gemma)
    to enable vision-language understanding and generation.
    
    Args:
        vision_config: Configuration dict for the vision encoder
        text_config: Configuration dict for the language model
        ignore_index: Label index to ignore in loss computation (default: -100)
        image_token_index: Special token ID representing image patches (default: 256000)
        vocab_size: Total vocabulary size including special tokens (default: 257152)
        projection_dim: Dimension for projecting vision features (default: 2048)
        hidden_size: Hidden size of the language model (default: 2048)
        pad_token_id: Padding token ID (default: None)
    """

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        
        # Loss computation configuration
        self.ignore_index = ignore_index
        
        # Special token for image embeddings
        self.image_token_index = image_token_index
        
        # Vocabulary configuration
        self.vocab_size = vocab_size
        
        # Projection from vision to language space
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        
        # Initialize sub-configurations
        self.vision_config = vision_config
        
        # Model architecture flag
        self.is_encoder_decoder = False
        
        # Padding token
        self.pad_token_id = pad_token_id

        # Instantiate vision configuration
        self.vision_config = SiglipVisionConfig(**vision_config)
        
        # Store text config dict
        self.text_config = text_config

        # Instantiate text configuration with padding token
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        
        # Update vocabulary size from text config
        self.vocab_size = self.text_config.vocab_size

        # Calculate number of image tokens based on patch size
        # For example: 224x224 image with 16x16 patches = (224/16)^2 = 196 tokens
        self.text_config.num_image_tokens = (
            self.vision_config.image_size // self.vision_config.patch_size
        ) ** 2
        
        # Set projection dimension in vision config
        self.vision_config.projection_dim = projection_dim