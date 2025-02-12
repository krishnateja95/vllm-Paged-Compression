import math
from vllm.utils import cdiv
from vllm.config import PagedEvictConfig


def get_num_required_blocks_after_prune_promt(q_len: int, paged_evict_config: PagedEvictConfig):
    """
    Get the required blocks after the prune prompt.
    """
    if q_len <= (paged_evict_config.initial_blocks + paged_evict_config.num_block_merge) * paged_evict_config.original_block_size:
        return cdiv(q_len, paged_evict_config.original_block_size), q_len 
    
    remainder_size = q_len % paged_evict_config.original_block_size
    end_first_slice_idx = paged_evict_config.initial_blocks * paged_evict_config.original_block_size
    num_first_slice_tokens = end_first_slice_idx
    
    end_middle_slice_idx = q_len - paged_evict_config.num_block_merge * paged_evict_config.original_block_size - remainder_size
    num_middle_slice_tokens = end_middle_slice_idx - end_first_slice_idx
    
    num_last_slice_tokens = q_len - end_middle_slice_idx
        
    if paged_evict_config.prompt_evict_method == "streamingLLM":  
        if paged_evict_config.cache_prune_type == "percentage":
            # Calculate the number of tokens after prunned the middle_slice_tokens
            pruned_middle_tokens = int(num_middle_slice_tokens/paged_evict_config.evict_freq) 
            # print(f"****1======prunning prompt with percentage using streamingLLM: pruned_middle_tokens={pruned_middle_tokens}")
            pruned_middle_tokens = paged_evict_config.original_block_size * (math.floor(pruned_middle_tokens/paged_evict_config.original_block_size))
            # print(f"****1======prunning prompt with percentage using streamingLLM: pruned_middle_tokens={pruned_middle_tokens}") 
            total_unpruned_tokens = num_first_slice_tokens + (num_middle_slice_tokens - pruned_middle_tokens) + num_last_slice_tokens
            # print(f"****1======prunning prompt with percentage using streamingLLM: total_unpruned_tokens={total_unpruned_tokens}")   
            return cdiv(total_unpruned_tokens, paged_evict_config.original_block_size), total_unpruned_tokens
        else:
            # Calculate the number of tokens after prunned the middle_slice_tokens
            middle_unpruned_tokens = paged_evict_config.cache_budget - \
                (paged_evict_config.initial_blocks * paged_evict_config.original_block_size) -\
                (paged_evict_config.num_block_merge * paged_evict_config.original_block_size)
            # print(f"****2======prunning prompt with budget using streamingLLM: middle_unpruned_tokens={middle_unpruned_tokens}")
                
            middle_unpruned_tokens = max(middle_unpruned_tokens, 0)
            total_unpruned_tokens = num_first_slice_tokens + middle_unpruned_tokens + num_last_slice_tokens
            # print(f"****2======prunning prompt with budget using streamingLLM: total_unpruned_tokens={total_unpruned_tokens}") 
            return cdiv(total_unpruned_tokens, paged_evict_config.original_block_size), total_unpruned_tokens
    else:
        if paged_evict_config.cache_prune_type == "percentage":
            # Calculate the number of tokens after prunned the middle_slice_tokens
            pruned_middle_tokens = int(num_middle_slice_tokens/paged_evict_config.evict_freq) 
            # print(f"****3======prunning prompt with percentage using {paged_evict_config.prompt_evict_method}: pruned_middle_tokens={pruned_middle_tokens}")
            pruned_middle_tokens = paged_evict_config.original_block_size * (math.floor(pruned_middle_tokens/paged_evict_config.original_block_size))
            # print(f"****3======prunning prompt with percentage using {paged_evict_config.prompt_evict_method}: pruned_middle_tokens={pruned_middle_tokens}")  
            total_unpruned_tokens = num_first_slice_tokens + (num_middle_slice_tokens - pruned_middle_tokens) + num_last_slice_tokens
            # print(f"****3======prunning prompt with percentage using {paged_evict_config.prompt_evict_method}: total_unpruned_tokens={total_unpruned_tokens}") 
            return cdiv(total_unpruned_tokens, paged_evict_config.original_block_size), total_unpruned_tokens
        else:
            # Calculate the number of tokens after prunned the middle_slice_tokens
            middle_unpruned_tokens = paged_evict_config.cache_budget - \
                (paged_evict_config.initial_blocks * paged_evict_config.original_block_size) -\
                (paged_evict_config.original_block_size * paged_evict_config.original_block_size)
            # print(f"****4======prunning prompt with budget using {paged_evict_config.prompt_evict_method}: middle_unpruned_tokens={middle_unpruned_tokens}")
                
            middle_unpruned_tokens = max(middle_unpruned_tokens, 0) 
            total_unpruned_tokens = num_first_slice_tokens + middle_unpruned_tokens + num_last_slice_tokens
            # print(f"****4======prunning prompt with budget using {paged_evict_config.prompt_evict_method}: total_unpruned_tokens={total_unpruned_tokens}")
            return cdiv(total_unpruned_tokens, paged_evict_config.original_block_size), total_unpruned_tokens
                
# get start and end block idx to to be removed after prune
def get_blocks_to_prune(paged_evict_config: PagedEvictConfig, seq_kv_len: int):    
    if paged_evict_config.decode_evict_method == "streamingLLM":
        if paged_evict_config.cache_prune_type == "percentage":
            if seq_kv_len <= (paged_evict_config.initial_blocks + paged_evict_config.num_block_merge) * paged_evict_config.original_block_size:
                s_block_id, e_block_id = -1, -1
                pruned_tokens = 0
            s_block_id = paged_evict_config.initial_blocks
            e_block_id = s_block_id + (paged_evict_config.num_block_merge - 1)
            pruned_tokens = (paged_evict_config.num_block_merge - 1) *paged_evict_config.original_block_size
        else:
            # For cache_budget case
            if seq_kv_len <= paged_evict_config.cache_budget:
                s_block_id, e_block_id = -1, -1
                pruned_tokens = 0
            else:    
                s_block_id = paged_evict_config.initial_blocks
                e_block_id = paged_evict_config.initial_blocks + 1
                pruned_tokens = paged_evict_config.original_block_size
        
    else:
        # our algorithm
        if paged_evict_config.cache_prune_type == "percentage":
            if seq_kv_len <= (paged_evict_config.initial_blocks + paged_evict_config.num_block_merge) * paged_evict_config.original_block_size:
                s_block_id, e_block_id = -1, -1
                pruned_tokens = 0
            s_block_id = paged_evict_config.initial_blocks + 1
            e_block_id = paged_evict_config.initial_blocks + paged_evict_config.num_block_merge
            pruned_tokens = (paged_evict_config.num_block_merge - 1) * paged_evict_config.original_block_size
            # print(f"pruned_tokens={pruned_tokens}")
        else:
            # For cache_budget case
            if seq_kv_len <= paged_evict_config.cache_budget:
                s_block_id, e_block_id = -1, -1
                pruned_tokens = 0
            else:
                # For cache_budget case, we will merge num_block_merge to num_block_merge - 1 blocks 
                s_block_id = paged_evict_config.initial_blocks + (paged_evict_config.num_block_merge - 1)
                e_block_id = paged_evict_config.initial_blocks + paged_evict_config.num_block_merge
                pruned_tokens = paged_evict_config.original_block_size
            
    return s_block_id, e_block_id, pruned_tokens
