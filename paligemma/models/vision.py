"""
Vision encoder components using SigLIP architecture.

SigLIP (Sigmoid Loss for Language Image Pre-training) is a vision transformer
that converts images into sequence of patch embeddings for the language model.

Architecture:
    Image -> Patch Embedding -> Transformer Encoder -> Normalized Features
"""

import torch
from torch import nn
from typing import Optional, Tuple

from config.config import SiglipVisionConfig


class SiglipVisionEmbeddings(nn.Module):
    """
    Convert images into patch embeddings with positional encodings.
    
    Process:
        1. Split image into fixed-size patches using convolution
        2. Flatten patches into sequence
        3. Add learned positional embeddings to each patch
    
    For a 224x224 image with 16x16 patches:
        - Number of patches = (224/16) * (224/16) = 14 * 14 = 196
        - Each patch becomes an embedding vector
    """
    
    def __init__(self, config: SiglipVisionConfig):
        """
        Initialize patch embedding layer and positional embeddings.
        
        Args:
            config: Configuration object with model dimensions and image specs
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # Convolutional layer acts as patch extraction + linear projection
        # Kernel size = stride = patch_size ensures non-overlapping patches
        # Input: [Batch, Channels=3, Height=224, Width=224]
        # Output: [Batch, Embed_Dim, Num_Patches_H, Num_Patches_W]
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,  # RGB = 3 channels
            out_channels=self.embed_dim,      # Project to model dimension
            kernel_size=self.patch_size,      # Size of each patch
            stride=self.patch_size,           # No overlap between patches
            padding="valid",                  # No padding added
        )

        # Calculate total number of patches
        # For 224x224 image with 16x16 patches: (224/16)^2 = 196
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        
        # Learnable positional embeddings for each patch position
        # These help the model understand spatial relationships
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        
        # Register position IDs as a buffer (saved with model but not trained)
        # Shape: [1, Num_Patches] containing [0, 1, 2, ..., 195]
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,  # Don't save in state_dict
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """
        Convert image pixels to patch embeddings with positions.
        
        Args:
            pixel_values: Input images [Batch_Size, Channels=3, Height, Width]
        
        Returns:
            Patch embeddings [Batch_Size, Num_Patches, Embed_Dim]
        
        Example shapes for 224x224 image:
            Input: [B, 3, 224, 224]
            After conv: [B, 768, 14, 14]
            After flatten: [B, 768, 196]
            After transpose: [B, 196, 768]
            After pos_emb: [B, 196, 768]
        """
        _, _, height, width = pixel_values.shape
        
        # Apply convolution to extract and project patches
        # Stride = kernel_size ensures non-overlapping patches
        # Output: [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        # where Num_Patches_H = height // patch_size
        #   and Num_Patches_W = width // patch_size
        patch_embeds = self.patch_embedding(pixel_values)
        
        # Flatten spatial dimensions into sequence dimension
        # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        #   -> [Batch_Size, Embed_Dim, Num_Patches]
        # where Num_Patches = Num_Patches_H * Num_Patches_W
        embeddings = patch_embeds.flatten(2)
        
        # Transpose to put sequence length before embedding dimension
        # [Batch_Size, Embed_Dim, Num_Patches]
        #   -> [Batch_Size, Num_Patches, Embed_Dim]
        # This matches transformer's expected input format
        embeddings = embeddings.transpose(1, 2)
        
        # Add positional embeddings to give model spatial awareness
        # Each patch gets a unique positional encoding added to its content
        # Broadcasting handles batch dimension automatically
        embeddings = embeddings + self.position_embedding(self.position_ids)
        
        # Output shape: [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings


class SiglipAttention(nn.Module):
    """
    Multi-headed self-attention mechanism.
    
    Allows each patch to attend to all other patches, learning
    relationships between different regions of the image.
    
    Formula: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
    """

    def __init__(self, config):
        """
        Initialize attention projection matrices.
        
        Args:
            config: Configuration with attention parameters
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5  # 1 / sqrt(head_dim) for scaled dot-product
        self.dropout = config.attention_dropout

        # Linear projections for queries, keys, and values
        # All project from embed_dim to embed_dim (then split across heads)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
        # Output projection to combine heads
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute multi-head self-attention.
        
        Args:
            hidden_states: Input embeddings [Batch_Size, Num_Patches, Embed_Dim]
        
        Returns:
            Tuple of (attention_output, attention_weights)
            - attention_output: [Batch_Size, Num_Patches, Embed_Dim]
            - attention_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project inputs to queries, keys, and values
        # Each: [Batch_Size, Num_Patches, Embed_Dim]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape and transpose for multi-head attention
        # Split embed_dim into (num_heads, head_dim)
        # [Batch_Size, Num_Patches, Embed_Dim]
        #   -> [Batch_Size, Num_Patches, Num_Heads, Head_Dim]
        #   -> [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        key_states = key_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        value_states = value_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        # Compute scaled dot-product attention scores
        # Q @ K^T: [Batch, Heads, Seq, Head_Dim] @ [Batch, Heads, Head_Dim, Seq]
        #   -> [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        # Scale by 1/sqrt(head_dim) to prevent softmax saturation
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        )

        # Validate attention weights shape
        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size "
                f"{(batch_size, self.num_heads, seq_len, seq_len)}, "
                f"but is {attn_weights.size()}"
            )

        # Apply softmax to get attention probabilities
        # Softmax along last dim (key dimension) normalizes attention per query
        # Use float32 for numerical stability then convert back
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        
        # Apply dropout for regularization (only during training)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )
        
        # Apply attention weights to values
        # [Batch, Heads, Seq, Seq] @ [Batch, Heads, Seq, Head_Dim]
        #   -> [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        attn_output = torch.matmul(attn_weights, value_states)

        # Validate attention output shape
        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size "
                f"{(batch_size, self.num_heads, seq_len, self.head_dim)}, "
                f"but is {attn_output.size()}"
            )
        
        # Transpose back to put sequence before heads
        # [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        #   -> [Batch_Size, Num_Patches, Num_Heads, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # Concatenate all heads by merging last two dimensions
        # [Batch_Size, Num_Patches, Num_Heads, Head_Dim]
        #   -> [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        
        # Final linear projection
        # [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipMLP(nn.Module):
    """
    Feed-forward network applied after attention.
    
    Two-layer MLP with GELU activation:
        hidden = GELU(Linear1(x))
        output = Linear2(hidden)
    
    Expands to intermediate_size then projects back to hidden_size.
    """
    
    def __init__(self, config):
        """
        Initialize MLP layers.
        
        Args:
            config: Configuration with hidden and intermediate sizes
        """
        super().__init__()
        self.config = config
        
        # First layer expands dimension
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        
        # Second layer projects back to original dimension
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply two-layer feedforward network.
        
        Args:
            hidden_states: Input [Batch_Size, Num_Patches, Embed_Dim]
        
        Returns:
            Output [Batch_Size, Num_Patches, Embed_Dim]
        """
        # Expand: [Batch_Size, Num_Patches, Embed_Dim]
        #   -> [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = self.fc1(hidden_states)
        
        # Apply GELU activation (smooth version of ReLU)
        # approximate="tanh" uses tanh approximation for speed
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        
        # Project back: [Batch_Size, Num_Patches, Intermediate_Size]
        #   -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class SiglipEncoderLayer(nn.Module):
    """
    Single transformer encoder layer with pre-norm architecture.
    
    Structure:
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
    
    Pre-norm (LayerNorm before operation) is more stable than post-norm.
    """
    
    def __init__(self, config: SiglipVisionConfig):
        """
        Initialize encoder layer components.
        
        Args:
            config: Configuration for attention and MLP
        """
        super().__init__()
        self.embed_dim = config.hidden_size
        
        # Self-attention mechanism
        self.self_attn = SiglipAttention(config)
        
        # Layer normalization before attention
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        
        # Feed-forward network
        self.mlp = SiglipMLP(config)
        
        # Layer normalization before MLP
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Process input through attention and feedforward with residual connections.
        
        Args:
            hidden_states: Input [Batch_Size, Num_Patches, Embed_Dim]
        
        Returns:
            Output [Batch_Size, Num_Patches, Embed_Dim]
        """
        # Store input for residual connection
        residual = hidden_states
        
        # Pre-norm: normalize before attention
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm1(hidden_states)
        
        # Self-attention block
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        
        # Residual connection: add input to attention output
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states
        
        # Store for next residual connection
        residual = hidden_states
        
        # Pre-norm: normalize before MLP
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm2(hidden_states)
        
        # Feed-forward network
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.mlp(hidden_states)
        
        # Residual connection: add input to MLP output
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states

        return hidden_states


class SiglipEncoder(nn.Module):
    """
    Stack of transformer encoder layers.
    
    Sequentially applies multiple encoder layers to progressively
    refine patch representations.
    """
    
    def __init__(self, config: SiglipVisionConfig):
        """
        Initialize encoder stack.
        
        Args:
            config: Configuration specifying number of layers
        """
        super().__init__()
        self.config = config
        
        # Create list of encoder layers
        # Each layer is identical in architecture but has separate parameters
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """
        Pass embeddings through all encoder layers.
        
        Args:
            inputs_embeds: Patch embeddings [Batch_Size, Num_Patches, Embed_Dim]
        
        Returns:
            Encoded features [Batch_Size, Num_Patches, Embed_Dim]
        """
        hidden_states = inputs_embeds

        # Sequentially apply each encoder layer
        for encoder_layer in self.layers:
            # Each layer takes and returns same shape
            # [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    """
    Complete vision transformer: embeddings + encoder + normalization.
    
    Pipeline:
        Images -> Patch Embeddings -> Transformer Layers -> Layer Norm -> Features
    """
    
    def __init__(self, config: SiglipVisionConfig):
        """
        Initialize full vision transformer.
        
        Args:
            config: Complete vision model configuration
        """
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        # Convert images to patch embeddings
        self.embeddings = SiglipVisionEmbeddings(config)
        
        # Stack of transformer encoder layers
        self.encoder = SiglipEncoder(config)
        
        # Final layer normalization for output stability
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features from images.
        
        Args:
            pixel_values: Input images [Batch_Size, Channels, Height, Width]
        
        Returns:
            Visual features [Batch_Size, Num_Patches, Embed_Dim]
        """
        # Convert pixels to patch embeddings with positions
        # [Batch_Size, Channels, Height, Width]
        #   -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.embeddings(pixel_values)

        # Process through transformer encoder layers
        # [Batch_Size, Num_Patches, Embed_Dim]
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)

        # Apply final normalization
        # [Batch_Size, Num_Patches, Embed_Dim]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state


class SiglipVisionModel(nn.Module):
    """
    Top-level vision model wrapper.
    
    Wraps SiglipVisionTransformer for consistent interface.
    """

    def __init__(self, config: SiglipVisionConfig):
        """
        Initialize vision model.
        
        Args:
            config: Vision model configuration
        """
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> torch.Tensor:
        """
        Forward pass through vision model.
        
        Args:
            pixel_values: Images [Batch_Size, Channels, Height, Width]
        
        Returns:
            Visual features [Batch_Size, Num_Patches, Embed_Dim]
        """
        # Delegate to underlying transformer
        # [Batch_Size, Channels, Height, Width]
        #   -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values)