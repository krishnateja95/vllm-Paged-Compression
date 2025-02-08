import torch
import torch.nn.functional as F
import math 

class KVCachePruner:
    def __init__(self, cache_prune_type, prompt_evict_method, decode_evict_method, block_size, evict_size, 
                 cache_budget, initial_blocks, num_block_merge):
        self.cache_prune_type = cache_prune_type
        self.prompt_evict_method = prompt_evict_method
        self.decode_evict_method = decode_evict_method
        self.orig_block_size = block_size
        if self.cache_prune_type == "percentage":
            self.evict_size = evict_size
            self.compressed_block_size = self.orig_block_size - self.evict_size
            self.compression_rate = int(self.orig_block_size / self.compressed_block_size)
        else:
            self.cache_budget = cache_budget
            self.compression_rate = num_block_merge
            
        self.initial_blocks = initial_blocks # This represents the number of compressed blocks
        self.num_block_merge = num_block_merge
        
    def cosine_similarity(self, tensor_a, tensor_b):
        # (TODO): check the shape of tensor_a and tensor_b
        a_norm = F.normalize(tensor_a, dim=-1) # dim=-1 means normalize along the last dimension (the head_size in our case)
        b_norm = F.normalize(tensor_b, dim=-1)
        return torch.sum(a_norm * b_norm, dim=-1)

    def key_l1(self, key_block):
        return torch.norm(key_block, p=1, dim = -1)

    def key_l2(self, key_block):
        return torch.norm(key_block, p=2, dim = -1)

    def value_l1(self, value_block):
        return torch.norm(value_block, p=1, dim = -1)    
    
    def value_l2(self, value_block):
        return torch.norm(value_block, p=2, dim = -1)
    
    def inverse_key_l1(self, key_block, value_block):
        return torch.div(1, self.key_l1(key_block) + 1e-8)
    
    def inverse_key_l2(self, key_block, value_block):
        return torch.div(1, self.key_l2(key_block) + 1e-8)
    
    def key_l1_div_value_l1(self, key_block, value_block):
        return torch.div(self.key_l2(key_block), self.value_l1(value_block) + 1e-8)
    
    def key_l1_div_value_l2(self, key_block, value_block):
        return torch.div(self.key_l1(key_block), self.value_l2(value_block) + 1e-8)
    
    def key_l2_div_value_l1(self, key_block, value_block):
        return torch.div(self.key_l2(key_block), self.value_l1(value_block) + 1e-8)
    
    def key_l2_div_value_l2(self, key_block, value_block):
        return torch.div(self.key_l2(key_block), self.value_l2(value_block) + 1e-8)

    def value_l1_div_key_l1(self, key_block, value_block):
        return torch.div(self.value_l1(key_block), self.key_l1(value_block) + 1e-8)
        
    def value_l1_div_key_l2(self, key_block, value_block):
        return torch.div(self.value_l1(key_block), self.key_l2(value_block) + 1e-8)

    def value_l2_div_key_l1(self, key_block, value_block):
        return torch.div(self.value_l2(key_block), self.key_l1(value_block) + 1e-8)
    
    def value_l2_div_key_l2(self, key_block, value_block):
        return torch.div(self.value_l2(key_block), self.key_l2(value_block) + 1e-8)

    def value_l1_plus_key_l1(self, key_block, value_block):
        return self.value_l1(key_block) + self.key_l1(value_block)
    
    def value_l2_plus_key_l2(self, key_block, value_block):
        return self.value_l2(key_block) + self.key_l2(value_block)

    def get_score(self, block_keys, block_values, evict_method):
        # Check the block_keys and block_values shape
        if evict_method == "cosine":
            scores = self.cosine_similarity(block_keys, block_values)

        elif evict_method == "key_l1":
            scores = self.key_l1(block_keys)

        elif evict_method == "key_l2":
            scores = self.key_l2(block_keys)

        elif evict_method == "value_l1":
            scores = self.value_l1(block_values)

        elif evict_method == "value_l2":
            scores = self.value_l2(block_values)

        elif evict_method == "key_l1_div_value_l1":
            scores = self.key_l1_div_value_l1(block_keys, block_values)
        
        elif evict_method == "key_l1_div_value_l2":
            scores = self.key_l1_div_value_l2(block_keys, block_values)

        elif evict_method == "key_l2_div_value_l1":
            scores = self.key_l2_div_value_l1(block_keys, block_values)
            
        elif evict_method == "key_l2_div_value_l2":
            scores = self.key_l2_div_value_l2(block_keys, block_values)

        elif evict_method == "value_l1_div_key_l1":
            scores = self.value_l1_div_key_l1(block_keys, block_values)
        
        elif evict_method == "value_l1_div_key_l2":
            scores = self.value_l1_div_key_l2(block_keys, block_values)

        elif evict_method == "value_l2_div_key_l1":
            scores = self.value_l2_div_key_l1(block_keys, block_values)
            
        elif evict_method == "value_l2_div_key_l2":
            scores = self.value_l2_div_key_l2(block_keys, block_values)

        elif evict_method == "value_l1_plus_key_l1":
            scores = self.value_l1_plus_key_l1(block_keys, block_values)
            
        elif evict_method == "value_l2_plus_key_l2":
            scores = self.value_l2_plus_key_l2(block_keys, block_values)

        elif evict_method == "streamingLLM":
            q_len, num_heads, head_dim = block_keys.shape
            scores = torch.arange(1, q_len + 1).view(-1, 1).expand(-1, num_heads)
        
        elif evict_method == "inverse_key_l1":
            scores = self.inverse_key_l1(block_keys, block_values)
        
        elif evict_method == "inverse_key_l2":
            scores = self.inverse_key_l2(block_keys, block_values)
        
        return scores


    def prune_prompt(self, key_tensor, value_tensor):
        q_len, num_heads, head_dim = key_tensor.shape
        
        if self.prompt_evict_method == "streamingLLM":
            if self.cache_prune_type == "percentage":
                if q_len <= (self.initial_blocks + self.compression_rate) * self.compressed_block_size:
                    return key_tensor, value_tensor
                
                remainder_size = q_len % self.compressed_block_size
                # Only the middle slice will be pruned
                end_idx_first_slice = self.initial_blocks * self.compressed_block_size
                first_slice = slice(0, end_idx_first_slice)
                
                end_idx_middle_slice = q_len - self.compression_rate * self.compressed_block_size - remainder_size
                middle_slice = slice(end_idx_first_slice, end_idx_middle_slice)
                
                last_slice = slice(end_idx_middle_slice, q_len)
                # print(f"**********1===remainder_size = {remainder_size}, q_len = {q_len}, first_slice = {first_slice}, middle_slice = {middle_slice}, last_slice = {last_slice}") 
                
                # Extract middle tokens and its size
                middle_key = key_tensor[middle_slice, :, :]
                middle_value = value_tensor[middle_slice, :, :]
                middle_tokens = middle_key.shape[0]
                
                # Calculate the number of tokens to prune from middle slice
                total_prune_tokens = middle_tokens - int(middle_tokens/self.compression_rate) 
                total_prune_tokens = self.compressed_block_size * (math.floor(total_prune_tokens/self.compressed_block_size))  

                # Middle slice after prunning
                middle_slice = slice(self.initial_blocks * self.compressed_block_size + total_prune_tokens, end_idx_middle_slice)
            else:
                assert self.cache_budget > (self.initial_blocks + self.compression_rate) * self.orig_block_size 
                if q_len <= self.cache_budget:
                    return key_tensor, value_tensor
                # For cache_budget case
                remainder_size = q_len % self.orig_block_size
                end_idx_first_slice = self.initial_blocks * self.orig_block_size
                first_slice = slice(0, end_idx_first_slice)
                end_idx_middle_slice = q_len - self.compression_rate * self.orig_block_size - remainder_size
                middle_slice = slice(end_idx_first_slice, end_idx_middle_slice)
                last_slice = slice(end_idx_middle_slice, q_len)
                # print(f"**********2===remainder_size = {remainder_size}, q_len = {q_len}, first_slice = {first_slice}, middle_slice = {middle_slice}, last_slice = {last_slice}") 
                
                # Extract middle tokens and its size
                middle_key = key_tensor[middle_slice, :, :]
                middle_value = value_tensor[middle_slice, :, :]
                # print(f"middle_key.shape: {middle_key.shape}, middle_value: {middle_value.shape}")
                middle_tokens = middle_key.shape[0]
                
                middle_pruned_tokens = self.cache_budget - (self.initial_blocks * self.orig_block_size) - (self.compression_rate * self.orig_block_size)
                total_prune_tokens = middle_tokens - middle_pruned_tokens
                total_prune_tokens = 0 if total_prune_tokens < 0 else total_prune_tokens
        
                # Middle slice after prunning
                middle_slice = slice(self.initial_blocks * self.orig_block_size + total_prune_tokens, end_idx_middle_slice) 
        
            # Rejoin the tokens from the first slice, prunned_middle_key, and last slice
            rejoined_key = torch.cat([
                key_tensor[first_slice, :, :],
                key_tensor[middle_slice, :, :],
                key_tensor[last_slice, :, :]
            ], dim=0)
            rejoined_value = torch.cat([
                value_tensor[first_slice, :, :],
                value_tensor[middle_slice, :, :],
                value_tensor[last_slice, :, :]
            ], dim=0)
            
            return rejoined_key, rejoined_value 
        else:
            if self.cache_prune_type == "percentage":
                if q_len <= (self.initial_blocks + self.compression_rate) * self.compressed_block_size:
                    return key_tensor, value_tensor
                 
                remainder_size = q_len % self.compressed_block_size
                end_idx_first_slice = self.initial_blocks * self.compressed_block_size
                first_slice = slice(0, end_idx_first_slice)
                
                end_idx_middle_slice = q_len - self.compression_rate * self.compressed_block_size - remainder_size
                middle_slice = slice(end_idx_first_slice, end_idx_middle_slice)

                last_slice = slice(end_idx_middle_slice, q_len)
                # print(f"**********remainder_size = {remainder_size}, q_len = {q_len}, first_slice = {first_slice}, middle_slice = {middle_slice}, last_slice = {last_slice}")
                
                # Extract middle tokens and its keys
                middle_key = key_tensor[middle_slice, :, :]
                middle_value = value_tensor[middle_slice, :, :]
                middle_tokens = middle_key.shape[0]
                
                total_prune_tokens = middle_tokens - int(middle_tokens/self.compression_rate) 
                total_prune_tokens = self.compressed_block_size * (math.floor(total_prune_tokens/self.compressed_block_size))  
                
            else:
                # For cache_budget case
                assert self.cache_budget > (self.initial_blocks + self.compression_rate) * self.orig_block_size 
                if q_len <= self.cache_budget:
                    return key_tensor, value_tensor
                
                remainder_size = q_len % self.orig_block_size 
                end_idx_first_slice = self.initial_blocks * self.orig_block_size 
                first_slice = slice(0, end_idx_first_slice)
                
                end_idx_middle_slice = q_len - self.compression_rate * self.orig_block_size - remainder_size
                middle_slice = slice(end_idx_first_slice, end_idx_middle_slice)
                
                last_slice = slice(end_idx_middle_slice, q_len)
                
                middle_key = key_tensor[middle_slice, :, :]
                middle_value = value_tensor[middle_slice, :, :]
                middle_tokens = middle_key.shape[0]
                
                middle_pruned_tokens = self.cache_budget - (self.initial_blocks * self.orig_block_size) - (self.compression_rate * self.orig_block_size)
                total_prune_tokens = middle_tokens - middle_pruned_tokens 
                total_prune_tokens = 0 if total_prune_tokens < 0 else total_prune_tokens
                
            # Get scores for middle tokens
            scores = self.get_score(middle_key, middle_value, self.prompt_evict_method)

            # Evict least N elements, select the indices along dim=0
            _, least_indices = torch.topk(scores, k=total_prune_tokens, largest=False, dim=0)

            # Create a mask for pruning
            mask = torch.ones_like(scores, dtype=torch.bool)
            mask.scatter_(0, least_indices, False)

            # Prune middle tokens, middle_key[mask] will drop the data based on the mask, only keep the data where mask is True
            pruned_middle_key = middle_key[mask].view(-1, num_heads, head_dim)
            pruned_middle_value = middle_value[mask].view(-1, num_heads, head_dim)

            # Rejoin the tokens from the first slice, prunned_middle_key, and last slice
            rejoined_key = torch.cat([
                key_tensor[first_slice, :, :],
                pruned_middle_key,
                key_tensor[last_slice, :, :]
            ], dim=0)
            rejoined_value = torch.cat([
                value_tensor[first_slice, :, :],
                pruned_middle_value,
                value_tensor[last_slice, :, :]
            ], dim=0)
            
            return rejoined_key, rejoined_value
        
    def get_blocks_to_prune_and_merge_decode(self, seq_kv_len):
        if self.decode_evict_method == "streamingLLM":
            if self.cache_prune_type == "percentage":
                assert self.evict_size % self.compressed_block_size == 0
                s_block_id = self.initial_blocks
                e_block_id = s_block_id + int(self.evict_size / self.compressed_block_size)
                prune_tokens = self.evict_size
            else:
                # For cache_budget case
                if seq_kv_len <= self.cache_budget:
                    s_block_id, e_block_id = -1, -1 # represent no blocks to prune
                    prune_tokens = 0
                else:
                    s_block_id = self.initial_blocks
                    e_block_id = self.initial_blocks + 1
                    prune_tokens = self.orig_block_size
        else:
            if self.cache_prune_type == "percentage":
                s_block_id = self.initial_blocks
                e_block_id = self.initial_blocks + int(self.orig_block_size / self.compressed_block_size)
                prune_tokens = self.evict_size
            else:
                # For cache_budget case
                if seq_kv_len <= self.cache_budget:
                    s_block_id, e_block_id = -1, -1 # No blocks to prune
                    prune_tokens = 0
                else:
                    s_block_id = self.initial_blocks
                    e_block_id = self.initial_blocks + self.num_block_merge
                    prune_tokens = self.orig_block_size
            
        return s_block_id, e_block_id, prune_tokens
          
    def prune_oldest_block(self, key_tensor, value_tensor):
        if self.decode_evict_method== "streamingLLM":
            # For vLLM, we just need to remove the block from the block_table, so do nothing here
            return None, None
        else:
            q_len, num_heads, head_dim = key_tensor.shape
            scores = self.get_score(key_tensor, value_tensor, self.decode_evict_method)
            # print(f"********scores.shape = {scores.shape}")
            if self.cache_prune_type == "percentage":
                prune_tokens = self.evict_size
            else:
                prune_tokens = self.orig_block_size
            # print(f"********Number of tokens to prune = {prune_tokens}")
            _, least_indices = torch.topk(scores, k=prune_tokens, largest=False, dim=0)

            mask = torch.ones_like(scores, dtype=torch.bool)
            mask.scatter_(0, least_indices, False)

            # print(f"mask.shape = {mask.shape}, key_tensor[mask].shape = {key_tensor[mask].shape}")
            pruned_key = key_tensor[mask].view(-1, num_heads, head_dim)
            pruned_value = key_tensor[mask].view(-1, num_heads, head_dim)
            
            return pruned_key, pruned_value        
    

    
if __name__ == "__main__":
    num_heads = 4
    head_dim = 64

    q_len = 8192
    block_size = 32
    evict_size = 16
    initial_blocks = 1
    prompt_evict_method = "streamingLLM"
    decode_evit_method = "value_l2"

    cache_type   = "budget"
    cache_budget = 128
    num_block_merge = 2
    
    pruner = KVCachePruner(cache_type, prompt_evict_method, decode_evit_method, block_size, evict_size, cache_budget, initial_blocks, num_block_merge)

    keys = torch.randn(q_len, num_heads, head_dim)
    values = torch.randn(q_len, num_heads, head_dim)

    print("before prompt blocks pruning", keys.shape, values.shape)
    keys, values = pruner.prune_prompt(keys, values)
    print("after prompt blocks pruning", keys.shape, values.shape)
    print()

    # keys = torch.randn(bsz, num_heads, q_len, head_dim)
    # values = torch.randn(bsz, num_heads, q_len, head_dim)

    # print("before recent block pruning", keys.shape, values.shape)
    # keys, values = pruner.prune_recent_block(keys, values)
    # print("after recent block pruning", keys.shape, values.shape)
    