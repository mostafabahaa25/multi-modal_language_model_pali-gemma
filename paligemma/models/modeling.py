"""
PaliGemma multimodal model combining vision and language.

PaliGemma architecture:
    Image -> Vision Encoder -> Projector -> |
                                            | -> Merged Embeddings -> Language Model -> Text
    Text -> Token Embeddings ------------> |
"""

import torch
from torch import nn
from typing import Optional, Tuple

from config.config import PaliGemmaConfig
from vision import SiglipVisionModel
from language import GemmaForCausalLM
from helper.utils import KVCache


class PaliGemmaMultiModalProjector(nn.Module):
    """
    Project vision encoder outputs to language model dimension.
    
    This linear layer bridges the vision and language modalities,
    mapping from vision hidden size to language hidden size.
    """
    
    def __init__(self, config: PaliGemmaConfig):
        """
        Initialize projection layer.
        
        Args:
            config: PaliGemma configuration with vision and text specs
        """
        super().__init__()
        
        # Linear projection with bias
        # Maps from vision encoder dimension to language model dimension
        self.linear = nn.Linear(
            config.vision_config.hidden_size,      # Input: vision features
            config.vision_config.projection_dim,   # Output: language-compatible features
            bias=True
        )

    def forward(self, image_features):
        """
        Project image features to language model space.
        
        Args:
            image_features: Vision features [Batch_Size, Num_Patches, Vision_Hidden_Size]
        
        Returns:
            Projected features [Batch_Size, Num_Patches, Language_Hidden_Size]
        """
        # Apply linear transformation
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Projection_Dim]
        hidden_states = self.linear(image_features)
        return hidden_states


class PaliGemmaForConditionalGeneration(nn.Module):
    """
    Complete PaliGemma model for vision-language tasks.
    
    Architecture:
        1. Vision encoder processes images into patch embeddings
        2. Multimodal projector maps vision features to language space
        3. Vision and text embeddings are merged
        4. Language model generates text conditioned on both modalities
    
    This enables tasks like:
        - Image captioning
        - Visual question answering
        - Object detection with text descriptions
    """
    
    def __init__(self, config: PaliGemmaConfig):
        """
        Initialize PaliGemma model.
        
        Args:
            config: Complete model configuration
        """
        super().__init__()
        self.config = config
        
        # Vision encoder (SigLIP)
        self.vision_tower = SiglipVisionModel(config.vision_config)
        
        # Projection from vision to language space
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        
        self.vocab_size = config.vocab_size

        # Language model (Gemma)
        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        # Padding token ID
        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )

    def tie_weights(self):
        """
        Tie embedding and language modeling head weights.
        
        This is called after model initialization to share weights
        between input embeddings and output projection.
        """
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ):
        """
        Merge image features with text token embeddings.
        
        This function:
            1. Scales image features by 1/sqrt(hidden_size)
            2. Creates a combined embedding sequence
            3. Places image features at <image> token positions
            4. Places text embeddings at text token positions
            5. Zeros out padding positions
            6. Creates appropriate attention mask
            7. Computes position IDs
        
        Args:
            image_features: Projected vision features [Batch, Num_Patches, Hidden_Size]
            inputs_embeds: Text token embeddings [Batch, Seq_Len, Hidden_Size]
            input_ids: Token IDs [Batch, Seq_Len]
            attention_mask: Original attention mask [Batch, Seq_Len]
            kv_cache: Optional KV cache for generation
        
        Returns:
            Tuple of (final_embedding, causal_mask, position_ids)
        """
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        
        # Scale image features (Gemma-specific normalization)
        # Divide by sqrt(hidden_size) for stability
        # [Batch_Size, Num_Patches, Hidden_Size]
        scaled_image_features = image_features / (self.config.hidden_size**0.5)

        # Initialize final embedding tensor (all zeros initially)
        # Will be filled with text and image embeddings
        # [Batch_Size, Seq_Len, Hidden_Size]
        final_embedding = torch.zeros(
            batch_size, sequence_length, embed_dim, 
            dtype=inputs_embeds.dtype, 
            device=inputs_embeds.device
        )
        
        # Create masks to identify different token types
        # Text mask: True for regular text tokens (not image, not padding)
        # [Batch_Size, Seq_Len]
        text_mask = (input_ids != self.config.image_token_index) & (
            input_ids != self.pad_token_id
        )
        
        # Image mask: True for image token positions
        # [Batch_Size, Seq_Len]
        image_mask = input_ids == self.config.image_token_index
        
        # Padding mask: True for padding tokens
        # [Batch_Size, Seq_Len]
        pad_mask = input_ids == self.pad_token_id

        # Expand masks to match embedding dimension for broadcasting
        # [Batch_Size, Seq_Len] -> [Batch_Size, Seq_Len, Hidden_Size]
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Insert text embeddings at text token positions
        # torch.where(condition, true_value, false_value)
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        
        # Insert image embeddings at image token positions
        # masked_scatter is used because image sequence length != final sequence length
        # It scatters scaled_image_features into positions where image_mask_expanded is True
        final_embedding = final_embedding.masked_scatter(
            image_mask_expanded, scaled_image_features
        )
        
        # Zero out padding positions
        final_embedding = torch.where(
            pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding
        )

        #### CREATE CAUSAL ATTENTION MASK ####
        
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min  # Large negative value for masked positions
        q_len = inputs_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            # Prefill phase: processing the entire prompt
            # No masking needed because we're not doing causal prediction yet
            # Each token can attend to itself and all previous tokens
            # Fill with zeros (no masking)
            # [Batch_Size, Q_Len, Q_Len]
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # Generation phase: generating one token at a time
            # Query is just the new token (q_len = 1)
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            
            # New token can attend to all previous tokens in cache
            # No masking needed in generation phase
            # [Batch_Size, 1, KV_Len]
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # Add head dimension for multi-head attention
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, 1, Q_Len, KV_Len]
        # The 1 will broadcast across all attention heads
        causal_mask = causal_mask.unsqueeze(1)

        #### COMPUTE POSITION IDs ####
        
        if kv_cache is not None and kv_cache.num_items() > 0:
            # Generation phase: position is last position in cache + 1
            # Cumulative sum of attention mask gives position indices
            # We take the last position for the new token
            # [Batch_Size, Seq_Len] -> [Batch_Size]
            position_ids = attention_mask.cumsum(-1)[:, -1]
            
            # Ensure 2D shape [Batch_Size, 1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Prefill phase: compute positions for all tokens
            # Position = cumulative sum of attention mask
            # Masked tokens (attention_mask=0) get position 1
            # [Batch_Size, Seq_Len]
            position_ids = (attention_mask.cumsum(-1)).masked_fill_(
                (attention_mask == 0), 1
            ).to(device)

        return final_embedding, causal_mask, position_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> dict:
        """
        Forward pass through PaliGemma model.
        
        Args:
            input_ids: Token IDs including image tokens [Batch_Size, Seq_Len]
            pixel_values: Image pixels [Batch_Size, Channels, Height, Width]
            attention_mask: Attention mask [Batch_Size, Seq_Len]
            kv_cache: KV cache for autoregressive generation
        
        Returns:
            Dictionary containing:
                - logits: Next token predictions [Batch_Size, Seq_Len, Vocab_Size]
                - kv_cache: Updated cache (if provided)
        
        Process:
            1. Extract text token embeddings
            2. Process image through vision encoder
            3. Project image features to language space
            4. Merge image and text embeddings
            5. Generate text with language model
        """
        # Validate input (current implementation doesn't support padding)
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # Step 1: Get text token embeddings from language model
        # [Batch_Size, Seq_Len] -> [Batch_Size, Seq_Len, Hidden_Size]
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Step 2: Process images through vision encoder
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Vision_Hidden_Size]
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        
        # Step 3: Project vision features to language model dimension
        # [Batch_Size, Num_Patches, Vision_Hidden_Size] -> [Batch_Size, Num_Patches, Hidden_Size]
        image_features = self.multi_modal_projector(selected_image_feature)

        # Step 4: Merge text and image embeddings into unified sequence
        # Returns:
        #   - inputs_embeds: Combined embeddings [Batch_Size, Seq_Len, Hidden_Size]
        #   - attention_mask: Causal mask [Batch_Size, 1, Q_Len, KV_Len]
        #   - position_ids: Position indices [Batch_Size, Seq_Len]
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids, attention_mask, kv_cache
        )

        # Step 5: Generate text with language model
        # Process merged embeddings through Gemma decoder
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs