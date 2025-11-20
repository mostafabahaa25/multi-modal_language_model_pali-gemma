"""
Utility classes and functions for PaliGemma model.

This module provides:
- KVCache: Key-Value caching for efficient autoregressive generation
- Sampling utilities: Top-p (nucleus) sampling for diverse text generation
"""

import torch
from typing import List, Tuple


class KVCache:
    """
    Key-Value cache for efficient transformer inference.
    
    During autoregressive generation, we compute attention over all previous tokens.
    Instead of recomputing keys and values for past tokens, we cache them.
    
    Structure:
        - key_cache: List of key tensors, one per layer
        - value_cache: List of value tensors, one per layer
        - Each tensor shape: [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
    
    Benefits:
        - Reduces computation from O(n²) to O(n) for sequence generation
        - Enables faster inference for long sequences
    """

    def __init__(self) -> None:
        """Initialize empty cache lists for keys and values."""
        # Store keys for each transformer layer
        self.key_cache: List[torch.Tensor] = []
        # Store values for each transformer layer
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) -> int:
        """
        Get the current sequence length stored in cache.
        
        Returns:
            int: Number of tokens cached (0 if cache is empty)
        """
        if len(self.key_cache) == 0:
            # Cache is empty, no tokens stored yet
            return 0
        else:
            # Return sequence length from the first layer's cache
            # Shape is [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            # We extract Seq_Len which is at index -2
            return self.key_cache[0].shape[-2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key and value states for a specific layer.
        
        Args:
            key_states: New key tensor [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            value_states: New value tensor [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            layer_idx: Index of the transformer layer (0-indexed)
        
        Returns:
            Tuple of (updated_keys, updated_values) containing all cached tokens
            including the new ones
        """
        if len(self.key_cache) <= layer_idx:
            # First time seeing this layer - initialize its cache
            # Store the key states directly
            self.key_cache.append(key_states)
            # Store the value states directly
            self.value_cache.append(value_states)
        else:
            # Layer already has cached states - append new tokens
            # Concatenate along sequence dimension (dim=-2)
            # Old shape: [B, H, old_seq_len, D]
            # New shape: [B, H, old_seq_len + new_seq_len, D]
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )

        # Return the complete cache for this layer (all tokens so far)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    Perform top-p (nucleus) sampling from a probability distribution.
    
    Top-p sampling selects from the smallest set of tokens whose cumulative
    probability exceeds the threshold p. This produces more diverse outputs
    than greedy decoding while avoiding low-probability tokens.
    
    Algorithm:
        1. Sort probabilities in descending order
        2. Compute cumulative sum of sorted probabilities
        3. Find cutoff where cumsum exceeds p
        4. Zero out probabilities below cutoff
        5. Renormalize and sample
    
    Args:
        probs: Probability distribution tensor [Batch_Size, Vocab_Size]
               Should be normalized (sum to 1.0)
        p: Cumulative probability threshold (typically 0.9 or 0.95)
           Controls diversity: lower p = more focused, higher p = more diverse
    
    Returns:
        Tensor of sampled token indices [Batch_Size, 1]
    
    Example:
        If p=0.9 and token probabilities are [0.5, 0.3, 0.15, 0.05]:
        - Cumsum: [0.5, 0.8, 0.95, 1.0]
        - Tokens with cumsum > 0.9: last two tokens are excluded
        - Sample from top 3 tokens with renormalized probs
    """
    # Step 1: Sort probabilities in descending order
    # probs_sort: sorted probabilities [Batch_Size, Vocab_Size]
    # probs_idx: original indices of sorted probabilities [Batch_Size, Vocab_Size]
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    
    # Step 2: Calculate cumulative sum of sorted probabilities
    # Shows how much total probability mass we've accumulated
    # Shape: [Batch_Size, Vocab_Size]
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    
    # Step 3: Create mask for tokens to exclude
    # Subtract current token prob to shift cumsum right by one position
    # This ensures we include enough tokens to exceed threshold p
    # Tokens where (cumsum - current_prob) > p are masked out
    # Shape: [Batch_Size, Vocab_Size], dtype: bool
    mask = probs_sum - probs_sort > p
    
    # Step 4: Zero out probabilities of excluded tokens
    # Apply mask to remove low-probability tail
    probs_sort[mask] = 0.0
    
    # Step 5: Renormalize probabilities to sum to 1.0
    # Divide each probability by the sum of remaining probabilities
    # in_place division for memory efficiency
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    
    # Step 6: Sample one token from the filtered distribution
    # multinomial samples indices according to probabilities
    # Shape: [Batch_Size, 1]
    next_token = torch.multinomial(probs_sort, num_samples=1)
    
    # Step 7: Map sampled index back to original vocabulary position
    # probs_idx contains original positions before sorting
    # gather retrieves the original index at the sampled position
    # Shape: [Batch_Size, 1]
    next_token = torch.gather(probs_idx, -1, next_token)
    
    return next_token