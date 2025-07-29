'''
A class to manage the l2 norm of all the blocks for the running requests.
This class is used when local/global eviction is enabled.
'''
from typing import Dict, List
import copy

class BlockL2NormManager:
    def __init__(self):
        self.reqid_mapping_blocktables: Dict[str, Dict[int, List[int]]] = None  # request_id -> block_tables mapping
        self.reqid_mapping_seqs: Dict[str, List[int]] = None  # request_id -> seq_ids mapping
        self.flat_sequence_map = [] # used to fetch the request_id and seq_id based on the seq_idx quickly
        self.reqid_mapping_block_l2norms: Dict[str, Dict[int, List[float]]] = None  # request_id -> block_l2norm mapping
        

    def update_reqs(self, reqid_mapping_blocktables: Dict[str, Dict[int, List[int]]]):
        ### Based on the blocktables of each request to init and update the their block_l2 norms
        ### Goal: add the new request and delete old request
        is_reqs_changed = False
        if self.reqid_mapping_block_l2norms is None:
            self.reqid_mapping_blocktables = reqid_mapping_blocktables
            self.reqid_mapping_seqs = {}
            self.reqid_mapping_block_l2norms = {} 
            for req_id, block_tables in reqid_mapping_blocktables.items():
                self.reqid_mapping_seqs[req_id] = list(block_tables.keys())  # Store seq_ids for the request
                self.reqid_mapping_block_l2norms[req_id] = {}
                for seq_id, block_table in block_tables.items():
                    self.reqid_mapping_block_l2norms[req_id][seq_id] = [0.0 * len(block_table)]  # Initialize with zeros
            is_reqs_changed = True
        else:
            # Update existing requests
            # print(f"update_reqs - pass to the function: reqid_mapping_blocktables={reqid_mapping_blocktables.keys()}")
            # compare the reqid_mapping_blocktables with the existing self.reqid_mapping_blocktables to if requests are changed
            if set(reqid_mapping_blocktables.keys()) != set(self.reqid_mapping_blocktables.keys()):
                is_reqs_changed = True
                
            self.reqid_mapping_blocktables = reqid_mapping_blocktables
            for req_id, block_tables in reqid_mapping_blocktables.items():
                if req_id not in self.reqid_mapping_block_l2norms:
                    self.reqid_mapping_seqs[req_id] = list(block_tables.keys()) 
                    self.reqid_mapping_block_l2norms[req_id] = {}
                for seq_id, block_table in block_tables.items():
                    if seq_id not in self.reqid_mapping_block_l2norms[req_id]:
                        self.reqid_mapping_block_l2norms[req_id][seq_id] = [0.0] * len(block_table)
            
            # print(f"update_reqs: reqid_mapping_blocktables={reqid_mapping_blocktables.keys()}, self.reqid_mapping_blocktables={self.reqid_mapping_blocktables.keys()}")
            # print(f"update_reqs: reqid_mapping_block_l2norms={self.reqid_mapping_block_l2norms.keys()}")
            # Remove old requests
            removed_reqs = [req_id for req_id in self.reqid_mapping_block_l2norms.keys() if req_id not in reqid_mapping_blocktables]
            # print(f"to be removed reqs: {removed_reqs}")
            for req_id in removed_reqs:
                del self.reqid_mapping_block_l2norms[req_id]
                del self.reqid_mapping_seqs[req_id]
            # print(f"update_reqs - after: reqid_mapping_seqs={self.reqid_mapping_seqs.keys()}")
                    
        if is_reqs_changed:
            ### reset the flat_sequence_map based on the updated reqid_mapping_seqs
            self.flat_sequence_map = []
            for req_id, seq_ids in self.reqid_mapping_seqs.items():
                for seq_id in seq_ids:
                    self.flat_sequence_map.append((req_id, seq_id))
            # print(f"update_reqs - flat_sequence_map: {self.flat_sequence_map}")

    def update_block_l2norms(self, seq_idx: int, l2_norms: List[float]):
        # calculate the req_id based on the seq_idx
        target_req_id, target_seq_id = self.get_reqid_and_seqid(seq_idx)
        assert target_req_id is not None and target_seq_id is not None, "Invalid seq_idx for block l2 norm update"
        ## update the block l2 norms for the target request and seq
        self.reqid_mapping_block_l2norms[target_req_id][target_seq_id] = l2_norms

    def update_last_block_l2norm(self, seq_idx: int, l2_norm: float):
        target_req_id, target_seq_id = self.get_reqid_and_seqid(seq_idx)
        assert target_req_id is not None and target_seq_id is not None, "Invalid seq_idx for block l2 norm update"
        self.reqid_mapping_block_l2norms[target_req_id][target_seq_id].append(l2_norm)
        return self.reqid_mapping_block_l2norms[target_req_id][target_seq_id]
        
    def get_reqid_and_seqid(self, seq_idx: int):
        # if seq_idx >= len(self.flat_sequence_map):
        #     for req_id, seq_ids in self.reqid_mapping_seqs.items():
        #         print(f"raw reqid_mapping_seqs: req_id={req_id}, seq_ids={seq_ids}")
            
        #     for item in self.flat_sequence_map:
        #         print(f"flat_sequence_map: item={item}")
            
        assert seq_idx < len(self.flat_sequence_map), f"seq_idx {seq_idx} is out of range length {len(self.flat_sequence_map)}"
        target_req_id, target_seq_id = self.flat_sequence_map[seq_idx]
        
        return target_req_id, target_seq_id
    
    def get_req_block_tables(self, seq_idx: int):
        req_id, seq_id = self.get_reqid_and_seqid(seq_idx)
        assert req_id is not None and seq_id is not None, "Invalid seq_idx for block tables retrieval"
        return self.reqid_mapping_blocktables[req_id][seq_id]
    
    def get_updated_block_tables(self, seq_idx: int, rmv_idx: int, is_rmv_l2norm: bool = False): 
        req_id, seq_id = self.get_reqid_and_seqid(seq_idx)
        assert req_id is not None and seq_id is not None, f"Invalid seq_idx {seq_idx} for block tables retrieval"
        # block_tables = self.reqid_mapping_blocktables[req_id][seq_id]
        block_tables = copy.deepcopy(self.reqid_mapping_blocktables[req_id][seq_id])
        # print(f"rmv_idx={rmv_idx}, block_tables={block_tables}")
        del block_tables[rmv_idx]  # Remove the block at rmv_idx
        if is_rmv_l2norm:
            # remove the l2-norm of the block at rmv_idx when is_rmv_l2norm is True
            del self.reqid_mapping_block_l2norms[req_id][seq_id][rmv_idx]
        # return the updated block tables
        return block_tables