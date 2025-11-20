"""
Gemma language model components.

Gemma is a decoder-only transformer for text generation. It uses:
- Grouped Query Attention (GQA) for efficiency
- Rotary Position Embeddings (RoPE) for position encoding
- RMS Normalization instead of LayerNorm
- SwiGLU activation in MLP
"""

import torch
from torch import nn
import math
from typing import Optional, Tuple

from config.config import GemmaConfig
from helper.utils import KVCache


class GemmaRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    RMSNorm is simpler and faster than LayerNorm:
    - No mean centering (only scales by RMS)
    - No bias term
    - Formula: x * rsqrt(mean(x^2) + eps) * weight
    
    Used throughout Gemma for normalizing activations.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize RMSNorm layer.
        
        Args:
            dim: Dimension of input features to normalize
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        
        # Learnable scaling parameter (initialized to zero, then adds 1.0 in forward)
        # This is different from LayerNorm which initializes to 1.0
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        """
        Compute RMS normalization.
        
        Args:
            x: Input tensor of any shape [..., dim]
        
        Returns:
            Normalized tensor with same shape as input
        """
        # Compute root mean square: sqrt(mean(x^2))
        # rsqrt computes 1/sqrt(x) directly for efficiency
        # keepdim=True preserves dimension for broadcasting
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Apply RMS normalization with learned scaling.
        
        Args:
            x: Input tensor [..., dim]
        
        Returns:
            Normalized and scaled tensor [..., dim]
        
        Note: Gemma uses (x * w).to(dtype) while Llama uses x.to(dtype) * w
        """
        # Normalize in float32 for numerical stability
        output = self._norm(x.float())
        
        # Apply learned weight parameter
        # Weight is initialized at 0, so (1.0 + weight) starts at 1.0
        # This is Gemma's specific formulation
        output = output * (1.0 + self.weight.float())
        
        # Convert back to original dtype
        return output.type_as(x)


class GemmaRotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    RoPE encodes position by rotating query and key vectors in complex space.
    This provides:
    - Relative position information
    - Extrapolation to longer sequences
    - No learned parameters
    
    Formula: rotate(x, θ) where θ_i = base^(-2i/dim) * position
    """
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        """
        Initialize rotary embedding frequencies.
        
        Args:
            dim: Dimension of attention head (must be even)
            max_position_embeddings: Maximum sequence length
            base: Base for frequency calculation (typically 10000)
            device: Device to place tensors on
        """
        super().__init__()

        self.dim = dim  # Should be head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Calculate inverse frequencies for rotation angles
        # Formula: theta_i = base^(-2i/dim) for i = 0, 1, ..., dim//2
        # These determine the rotation speed for each dimension pair
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        )
        
        # Register as buffer (saved with model but not trained)
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        """
        Compute cos and sin values for rotary embeddings.
        
        Args:
            x: Input tensor [batch_size, num_heads, seq_len, head_dim]
            position_ids: Position indices [batch_size, seq_len]
            seq_len: Sequence length (unused, kept for compatibility)
        
        Returns:
            Tuple of (cos, sin) tensors [batch_size, seq_len, head_dim]
            These are used to rotate query and key vectors
        """
        # Ensure inv_freq is on same device as input
        self.inv_freq.to(x.device)
        
        # Expand inv_freq for batch dimension
        # [Head_Dim // 2] -> [Batch_Size, Head_Dim // 2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1
        )
        
        # Expand position_ids for multiplication
        # [Batch_Size, Seq_Len] -> [Batch_Size, 1, Seq_Len]
        position_ids_expanded = position_ids[:, None, :].float()
        
        # Handle device type for autocast
        device_type = x.device.type
        device_type = (
            device_type if isinstance(device_type, str) and device_type != "mps" 
            else "cpu"
        )
        
        # Compute rotation angles without autocast for precision
        with torch.autocast(device_type=device_type, enabled=False):
            # Matrix multiplication: [B, D//2, 1] @ [B, 1, S] -> [B, D//2, S]
            # Then transpose: [B, D//2, S] -> [B, S, D//2]
            # This computes theta = inv_freq * position for each position
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            
            # Concatenate frequencies to match head_dim
            # [B, S, D//2] -> [B, S, D] by repeating each frequency
            # This creates pairs for complex number representation
            emb = torch.cat((freqs, freqs), dim=-1)
            
            # Compute cos and sin of all rotation angles
            # These will be applied to query and key vectors
            # Shape: [Batch_Size, Seq_Len, Head_Dim]
            cos = emb.cos()
            sin = emb.sin()
        
        # Return in same dtype as input
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """
    Rotate half of the hidden dims of the input.
    
    This function is used to apply the imaginary part of the rotation.
    It rearranges elements to prepare for complex number multiplication.
    
    Args:
        x: Input tensor [..., dim]
    
    Returns:
        Tensor with pairs swapped and negated [..., dim]
    
    Example:
        Input: [x1, x2, x3, x4]
        Output: [-x3, -x4, x1, x2]
        
    This represents the imaginary component in: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    """
    # Split tensor in half along last dimension
    x1 = x[..., : x.shape[-1] // 2]  # First half
    x2 = x[..., x.shape[-1] // 2 :]  # Second half
    
    # Concatenate with second half negated and placed first
    # This implements the rotation in complex space
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    Apply rotary positional embeddings to query and key tensors.
    
    This implements the core RoPE operation: rotating vectors by position-dependent angles.
    The rotation is applied in pairs of dimensions (treating them as complex numbers).
    
    Args:
        q: Query tensor [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim]
        k: Key tensor [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        cos: Cosine values [Batch_Size, Seq_Len, Head_Dim]
        sin: Sine values [Batch_Size, Seq_Len, Head_Dim]
        unsqueeze_dim: Dimension to unsqueeze for broadcasting (default: 1 for heads)
    
    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as inputs
    
    Formula (per dimension pair):
        q' = q * cos + rotate_half(q) * sin
        
    This is equivalent to complex multiplication: q * e^(i*theta)
    """
    # Add head dimension for broadcasting
    # [B, S, D] -> [B, 1, S, D]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    # Apply rotation formula
    # Real part: q * cos, Imaginary part: rotate_half(q) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class GemmaMLP(nn.Module):
    """
    Gemma's feed-forward network using SwiGLU activation.
    
    SwiGLU (Swish-Gated Linear Unit) is defined as:
        SwiGLU(x) = (xW_gate * swish(x)) W_down
        where swish(x) = x * sigmoid(x) ≈ gelu(x)
    
    This provides better performance than standard FFN.
    """
    
    def __init__(self, config):
        """
        Initialize MLP layers.
        
        Args:
            config: Model configuration with hidden and intermediate sizes
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Three linear layers (no bias in Gemma)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        """
        Apply SwiGLU feed-forward network.
        
        Args:
            x: Input [Batch_Size, Seq_Len, Hidden_Size]
        
        Returns:
            Output [Batch_Size, Seq_Len, Hidden_Size]
        
        Computation:
            gate = GELU(gate_proj(x))
            up = up_proj(x)
            output = down_proj(gate * up)
        """
        # Compute gated activation and up-projection in one line
        # This is equivalent to the commented code below but more concise
        return self.down_proj(
            nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads to match number of query heads.
    
    This implements Grouped Query Attention (GQA):
    - Fewer KV heads than Q heads reduces memory
    - Each KV head is shared across multiple Q heads
    
    Args:
        hidden_states: Key or Value tensor [Batch, Num_KV_Heads, Seq, Head_Dim]
        n_rep: Number of times to repeat each head (num_q_heads // num_kv_heads)
    
    Returns:
        Repeated tensor [Batch, Num_Q_Heads, Seq, Head_Dim]
    
    Example:
        If 8 Q heads and 2 KV heads, n_rep=4
        Each KV head is repeated 4 times to match Q heads
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    
    # No repetition needed if n_rep is 1
    if n_rep == 1:
        return hidden_states
    
    # Expand KV heads: add new dimension and repeat
    # [B, KV_H, S, D] -> [B, KV_H, 1, S, D] -> [B, KV_H, n_rep, S, D]
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    
    # Reshape to merge repeated heads into head dimension
    # [B, KV_H, n_rep, S, D] -> [B, KV_H * n_rep, S, D]
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class GemmaAttention(nn.Module):
    """
    Grouped Query Attention with Rotary Position Embeddings.
    
    GQA uses fewer key/value heads than query heads for efficiency:
    - Standard MHA: num_heads KV heads
    - GQA: num_kv_heads < num_heads (e.g., 8 Q heads, 2 KV heads)
    - MQA: 1 KV head (extreme case)
    
    RoPE adds position information through rotation rather than learned embeddings.
    """

    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        """
        Initialize attention module.
        
        Args:
            config: Model configuration
            layer_idx: Index of this layer in the model (for KV cache)
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Attention configuration
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        
        # Number of query heads per KV head (for GQA)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True  # Decoder attention is always causal

        # Validate configuration
        assert self.hidden_size % self.num_heads == 0

        # Query projection (full number of heads)
        self.q_proj = nn.Linear(
            self.hidden_size, 
            self.num_heads * self.head_dim, 
            bias=config.attention_bias
        )
        
        # Key projection (reduced number of heads for GQA)
        self.k_proj = nn.Linear(
            self.hidden_size, 
            self.num_key_value_heads * self.head_dim, 
            bias=config.attention_bias
        )
        
        # Value projection (reduced number of heads for GQA)
        self.v_proj = nn.Linear(
            self.hidden_size, 
            self.num_key_value_heads * self.head_dim, 
            bias=config.attention_bias
        )
        
        # Output projection
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, 
            self.hidden_size, 
            bias=config.attention_bias
        )
        
        # Rotary position embedding
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute grouped query attention with rotary embeddings.
        
        Args:
            hidden_states: Input [Batch_Size, Seq_Len, Hidden_Size]
            attention_mask: Mask [Batch_Size, 1, Seq_Len_Q, Seq_Len_KV]
            position_ids: Position indices [Batch_Size, Seq_Len]
            kv_cache: Cache for autoregressive generation
        
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        bsz, q_len, _ = hidden_states.size()
        
        # Project to queries, keys, values
        # Q: [Batch_Size, Seq_Len, Num_Heads_Q * Head_Dim]
        query_states = self.q_proj(hidden_states)
        # K: [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]
        key_states = self.k_proj(hidden_states)
        # V: [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        # [Batch_Size, Seq_Len, Num_Heads, Head_Dim] -> [Batch_Size, Num_Heads, Seq_Len, Head_Dim]
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # Apply rotary position embeddings
        # cos, sin: [Batch_Size, Seq_Len, Head_Dim]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        
        # Rotate queries and keys
        # Output shapes preserved
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Update KV cache if provided
        if kv_cache is not None:
            key_states, value_states = kv_cache.update(
                key_states, value_states, self.layer_idx
            )

        # Repeat KV heads to match number of query heads (GQA)
        # [Batch, Num_KV_Heads, Seq, Head_Dim] -> [Batch, Num_Q_Heads, Seq, Head_Dim]
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Compute attention scores: Q @ K^T / sqrt(head_dim)
        # [Batch, Num_Heads_Q, Seq_Len_Q, Head_Dim] @ [Batch, Num_Heads_Q, Head_Dim, Seq_Len_KV]
        # -> [Batch_Size, Num_Heads_Q, Seq_Len_Q, Seq_Len_KV]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        # Apply attention mask (for causal masking and padding)
        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask

        # Apply softmax to get attention probabilities
        # [Batch_Size, Num_Heads_Q, Seq_Len_Q, Seq_Len_KV]
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        
        # Apply dropout
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        
        # Apply attention to values
        # [Batch, Heads, Seq_Q, Seq_KV] @ [Batch, Heads, Seq_KV, Head_Dim]
        # -> [Batch_Size, Num_Heads_Q, Seq_Len_Q, Head_Dim]
        attn_output = torch.matmul(attn_weights, value_states)

        # Validate output shape
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size "
                f"{(bsz, self.num_heads, q_len, self.head_dim)}, "
                f"but is {attn_output.size()}"
            )
        
        # Transpose and reshape to merge heads
        # [Batch, Heads, Seq, Head_Dim] -> [Batch, Seq, Heads, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # [Batch, Seq, Heads, Head_Dim] -> [Batch, Seq, Heads * Head_Dim]
        attn_output = attn_output.view(bsz, q_len, -1)
        
        # Final output projection
        # [Batch_Size, Seq_Len_Q, Hidden_Size]
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class GemmaDecoderLayer(nn.Module):
    """
    Single Gemma decoder layer with pre-norm architecture.
    
    Structure:
        x = x + Attention(RMSNorm(x))
        x = x + MLP(RMSNorm(x))
    """

    def __init__(self, config: GemmaConfig, layer_idx: int):
        """
        Initialize decoder layer.
        
        Args:
            config: Model configuration
            layer_idx: Index of this layer (for KV cache)
        """
        super().__init__()
        self.hidden_size = config.hidden_size

        # Attention and MLP modules
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
        self.mlp = GemmaMLP(config)
        
        # Pre-normalization layers
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        """
        Process input through attention and MLP with residual connections.
        
        Args:
            hidden_states: Input [Batch_Size, Seq_Len, Hidden_Size]
            attention_mask: Attention mask
            position_ids: Position IDs
            kv_cache: KV cache for generation
        
        Returns:
            Output [Batch_Size, Seq_Len, Hidden_Size]
        """
        # Store for residual connection
        residual = hidden_states
        
        # Pre-norm and attention
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        
        # Residual connection
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = residual + hidden_states

        # Store for next residual
        # [Batch_Size, Seq_Len, Hidden_Size]
        residual = hidden_states
        
        # Pre-norm and MLP
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        # Residual connection
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = residual + hidden_states

        return hidden_states


class GemmaModel(nn.Module):
    """
    Gemma transformer model (embeddings + decoder layers + norm).
    
    This is the core language model without the final language modeling head.
    """

    def __init__(self, config: GemmaConfig):
        """
        Initialize Gemma model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Token embedding table
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        
        # Stack of decoder layers
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        # Final normalization
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        """Return embedding layer (used by PaliGemma)."""
        return self.embed_tokens

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        """
        Forward pass through Gemma model.
        
        Args:
            attention_mask: Attention mask
            position_ids: Position IDs
            inputs_embeds: Pre-computed embeddings [Batch_Size, Seq_Len, Hidden_Size]
            kv_cache: KV cache for generation
        
        Returns:
            Hidden states [Batch_Size, Seq_Len, Hidden_Size]
        """
        # Use provided embeddings (for multimodal, these combine text and image)
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = inputs_embeds
        
        # Normalize embeddings (Gemma-specific)
        # Scale by sqrt(hidden_size) for training stability
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        # Pass through all decoder layers
        for decoder_layer in self.layers:
            # [Batch_Size, Seq_Len, Hidden_Size]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        # Final normalization
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.norm(hidden_states)

        return hidden_states


class GemmaForCausalLM(nn.Module):
    """
    Gemma model with language modeling head for text generation.
    
    Adds final linear layer to project hidden states to vocabulary logits.
    """

    def __init__(self, config):
        """
        Initialize Gemma for causal language modeling.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # Core transformer model
        self.model = GemmaModel(config)
        
        self.vocab_size = config.vocab_size
        
        # Language modeling head (projects to vocabulary)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        """Return embedding layer."""
        return self.model.embed_tokens

    def tie_weights(self):
        """
        Tie embedding and output projection weights.
        
        Weight tying reduces parameters and often improves performance.
        The embedding and output projection share the same weight matrix.
        """
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> dict:
        """
        Forward pass for language modeling.
        
        Args:
            attention_mask: Attention mask
            position_ids: Position IDs
            inputs_embeds: Input embeddings [Batch_Size, Seq_Len, Hidden_Size]
            kv_cache: KV cache for generation
        
        Returns:
            Dictionary containing:
                - logits: Vocabulary logits [Batch_Size, Seq_Len, Vocab_Size]
                - kv_cache: Updated cache (if provided)
        """
        # Get hidden states from transformer
        # [Batch_Size, Seq_Len, Hidden_Size]
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        hidden_states = outputs
        
        # Project to vocabulary
        # [Batch_Size, Seq_Len, Vocab_Size]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        # Prepare return dictionary
        return_data = {
            "logits": logits,
        }

        # Include updated cache if provided
        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache

        return return_data