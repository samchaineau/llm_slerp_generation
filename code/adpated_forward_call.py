from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from time import time 
import torch
from forward_slerp import merge_tokens

def run_merge(tokens : torch.Tensor, 
              cutoff : int, 
              starting_tokens :int,
              max_tokens_before_keeping_end :int,
              ending_tokens :int,
              model):
    """
    Run the forward pass of the Mistral model with token merging.

    Args:
        tokens (torch.Tensor): Input tokens.
        cutoff (int): Index of the layer where token merging is applied.
        starting_tokens (int) : Number of tokens to keep unmerged at the beginning of the sequence.
        max_tokens_before_keeping_end  (int) : Max number of tokens to reach before keeping ending tokens unmerged.
        ending_tokens (int) : Number of tokens to keep unmerged at the end.
        model: The Mistral model instance.

    Returns:
        tuple: Tuple containing logits and time taken for the forward pass.
    """
    start = time()
    embed = model.model.embed_tokens(tokens)
    merge_embed = embed.clone()
    bs, sl, _ = merge_embed.shape
    encoded = merge_embed
    
    attention_mask = _prepare_4d_causal_attention_mask(None,
                                                  (bs, sl),
                                                  encoded,
                                                  0,
                                                  sliding_window=model.config.sliding_window)
    
    for i, layer in enumerate(model.model.layers):
        encoded = layer(encoded, attention_mask=attention_mask)
        encoded = encoded[0]
        if i == cutoff:
            start = hidden_states[:,:starting_tokens,:]
            end = hidden_states[:,starting_tokens:,:]
            
            if end.shape[1] > max_tokens_before_keeping_end:
                middle = end[:,:-ending_tokens,:]
                end = end[:,-ending_tokens:,:]
                middle = merge_tokens(middle)
                middle = middle.half()
                hidden_states = torch.cat((start, middle, end), axis = 1)
                
            else:    
                end = merge_tokens(end)
                end = end.half()
                hidden_states = torch.cat((start, end), axis = 1)
            
            bs, sl, _ = hidden_states.shape
            attention_mask = _prepare_4d_causal_attention_mask(None,
                                                            (bs, sl),
                                                            hidden_states,
                                                            0,
                                                            sliding_window=model.config.sliding_window)
    
    normed = model.model.norm(encoded)
    logits = model.lm_head(normed)
    end = time()
    length = end - start
    del embed, merge_embed, encoded, att_masks, normed
    torch.cuda.empty_cache()
    return logits.cpu(), length / 60