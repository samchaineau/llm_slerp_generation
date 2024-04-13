from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from time import time 
import torch
from forward_slerp import merge_tokens

def run_merge(tokens, cutoff, model):
    """
    Run the forward pass of the Mistral model with token merging.

    Args:
        tokens (torch.Tensor): Input tokens.
        cutoff (int): Index of the layer where token merging is applied.
        model: The Mistral model instance.

    Returns:
        tuple: Tuple containing logits and time taken for the forward pass.
    """
    start = time()
    embed = model.model.embed_tokens(tokens)
    merge_embed = embed.clone()
    bs, sl, _ = merge_embed.shape
    encoded = merge_embed
    
    att_masks = _prepare_4d_causal_attention_mask(None,
                                                  (bs, sl),
                                                  encoded,
                                                  0,
                                                  sliding_window=model.config.sliding_window)
    
    for i, layer in enumerate(model.model.layers):
        encoded = layer(encoded, attention_mask=att_masks)
        encoded = encoded[0]
        if i == cutoff:
            encoded = merge_tokens(encoded)
            bs, sl, _ = encoded.shape
            start_new_att = time()
            att_masks = _prepare_4d_causal_attention_mask(None,
                                                          (bs, sl),
                                                          encoded,
                                                          0,
                                                          sliding_window=model.config.sliding_window)
    
    normed = model.model.norm(encoded)
    logits = model.lm_head(normed)
    end = time()
    length = end - start
    del embed, merge_embed, encoded, att_masks, normed
    torch.cuda.empty_cache()
    return logits.cpu(), length / 60