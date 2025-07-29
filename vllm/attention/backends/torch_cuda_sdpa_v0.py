""" Attention layer with torch scaled_dot_product_attention
    and PagedAttention."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type
import collections
import numpy as np

import torch
from torch.nn.functional import scaled_dot_product_attention

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata,
                                              AttentionMetadataBuilder,
                                              AttentionType)
from vllm.attention.backends.utils import (
    PAD_SLOT_ID, CommonAttentionState,
    is_block_tables_empty, compute_slot_mapping_start_idx, compute_slot_mapping)

from vllm.attention.ops.paged_attn import (PagedAttention,
                                           PagedAttentionMetadata)
from vllm.utils import async_tensor_h2d, make_tensor_with_pad, print_warning_once
from vllm.multimodal import MultiModalPlaceholderMap
from collections import defaultdict
from itertools import accumulate
import time

if TYPE_CHECKING:
    from vllm.worker.model_runner import (ModelInputForGPUBuilder,
                                          ModelInputForGPUWithSamplingMetadata)


from vllm.config import CacheConfig, PagedEvictConfig
from vllm.attention.kvcache_prunner import KVCachePruner
from vllm.attention.block_l2norm_manager import BlockL2NormManager
from vllm.logger import init_logger
import random
from vllm.model_executor.models.utils import extract_layer_index
from vllm.worker.tmp_cache_singleton import TmpCacheSingleton


logger = init_logger(__name__)

class TorchCUDASDPABackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "TORCH_SDPA"

    @staticmethod
    def get_impl_cls() -> Type["TorchCUDASDPABackendImpl"]:
        return TorchCUDASDPABackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return TorchCUDASDPAMetadata

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_builder_cls() -> Type["TorchCUDASDPAMetadataBuilder"]:
        return TorchCUDASDPAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(num_blocks, block_size,
                                                 num_kv_heads, head_size)
 
    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)

####################################################################################################
@dataclass
class TorchCUDASDPAMetadata(AttentionMetadata, PagedAttentionMetadata):
    """Metadata for TorchCUDASDPABackend.
    """
    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]

    # FIXME: It is for flash attn.
    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool

    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]] = None

    # FIXME: It is for flash attn.
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor] = None

    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor] = None

    # Maximum query length in the batch. None for decoding.
    max_query_len: Optional[int] = None
    # Max number of query tokens among request in the batch.
    max_decode_query_len: Optional[int] = None
    
    query_start_loc: Optional[torch.Tensor] = None

    # Self-attention prefill/decode metadata cache
    _cached_prefill_metadata: Optional["TorchCUDASDPAMetadata"] = None
    _cached_decode_metadata: Optional["TorchCUDASDPAMetadata"] = None

    # Begin encoder attn & enc/dec cross-attn fields...
    # Encoder sequence lengths representation
    encoder_seq_lens: Optional[List[int]] = None
    encoder_seq_lens_tensor: Optional[torch.Tensor] = None

    # Maximum sequence length among encoder sequences
    max_encoder_seq_len: Optional[int] = None

    # Number of tokens input to encoder
    num_encoder_tokens: Optional[int] = None

    # Cross-attention memory-mapping data structures: slot mapping
    # and block tables
    cross_slot_mapping: Optional[torch.Tensor] = None
    cross_block_tables: Optional[torch.Tensor] = None

    # # seq_kv_lens used for paged_evict
    seq_kv_lens: Optional[List[int]] = None
    
    # seq_kv_lens_tensor stored as a tensor used for paged_evict
    seq_kv_lens_tensor: Optional[torch.Tensor] = None
    
    # cloned_block_tables: Optional[torch.Tensor] = None
    
    # seq_kv_lens_before_prune: Optional[List[int]] = None
    
    reqid_mapping_blocktables: Dict[str, Dict[str, List[int]]] = None 

    def __post_init__(self):
        # Set during the execution of the first attention op.
        # It is a list because it is needed to set per prompt
        # when alibi slopes is used. It is because of the limitation
        # from xformer API.
        # will not appear in the __repr__ and __init__
        self.attn_bias: Optional[List[torch.Tensor]] = None
        self.encoder_attn_bias: Optional[List[torch.Tensor]] = None
        self.cross_attn_bias: Optional[List[torch.Tensor]] = None

    @property
    def is_all_encoder_attn_metadata_set(self):
        '''
        All attention metadata required for encoder attention is set.
        '''
        return ((self.encoder_seq_lens is not None)
                and (self.encoder_seq_lens_tensor is not None)
                and (self.max_encoder_seq_len is not None))

    @property
    def is_all_cross_attn_metadata_set(self):
        '''
        All attention metadata required for enc/dec cross-attention is set.

        Superset of encoder attention required metadata.
        '''
        return (self.is_all_encoder_attn_metadata_set
                and (self.cross_slot_mapping is not None)
                and (self.cross_block_tables is not None))

    @property
    def prefill_metadata(self) -> Optional["TorchCUDASDPAMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            # Recover cached prefill-phase attention
            # metadata structure
            return self._cached_prefill_metadata

        assert ((self.seq_lens is not None)
                or (self.encoder_seq_lens is not None))
        assert ((self.seq_lens_tensor is not None)
                or (self.encoder_seq_lens_tensor is not None))

        # Compute some attn_metadata fields which default to None
        query_start_loc = (None if self.query_start_loc is None else
                           self.query_start_loc[:self.num_prefills + 1])
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[:self.num_prefill_tokens])
        seq_lens = (None if self.seq_lens is None else
                    self.seq_lens[:self.num_prefills])
        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[:self.num_prefills])
        context_lens_tensor = (None if self.context_lens_tensor is None else
                               self.context_lens_tensor[:self.num_prefills])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[:self.num_prefills])
        
        seq_kv_lens = (None if self.seq_kv_lens is None else
                       self.seq_kv_lens[:self.num_prefills])
        
        seq_kv_lens_tensor = (None if self.seq_kv_lens_tensor is None else
                                self.seq_kv_lens_tensor[:self.num_prefills])
        
        # cloned_block_tables = (None if self.cloned_block_tables is None else
        #                        self.cloned_block_tables[:self.num_prefills])
        
        # seq_kv_lens_before_prune = (None if self.seq_kv_lens_before_prune is None else
        #                             self.seq_kv_lens_before_prune[:self.num_prefills])

        # Construct & cache prefill-phase attention metadata structure
        self._cached_prefill_metadata = TorchCUDASDPAMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=self.
            multi_modal_placeholder_index_maps,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=self.max_query_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
            query_start_loc=query_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=False,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables,
            seq_kv_lens=seq_kv_lens,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            # cloned_block_tables=cloned_block_tables,
            # seq_kv_lens_before_prune=seq_kv_lens_before_prune
            )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["TorchCUDASDPAMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            # Recover cached decode-phase attention
            # metadata structure
            return self._cached_decode_metadata
        assert ((self.seq_lens_tensor is not None)
                or (self.encoder_seq_lens_tensor is not None))

        # Compute some attn_metadata fields which default to None
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[self.num_prefill_tokens:])
        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[self.num_prefills:])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[self.num_prefills:])
        seq_kv_lens = (None if self.seq_kv_lens is None else
                       self.seq_kv_lens[self.num_prefills:])
        seq_kv_lens_tensor = (None if self.seq_kv_lens_tensor is None else
                                self.seq_kv_lens_tensor[self.num_prefills:])
        # cloned_block_tables = (None if self.cloned_block_tables is None else
        #                        self.cloned_block_tables[self.num_prefills:])
        # seq_kv_lens_before_prune = (None if self.seq_kv_lens_before_prune is None else
        #                             self.seq_kv_lens_before_prune[self.num_prefills:])

        # Construct & cache decode-phase attention metadata structure
        self._cached_decode_metadata = TorchCUDASDPAMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=None,
            seq_lens_tensor=seq_lens_tensor,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            block_tables=block_tables,
            use_cuda_graph=self.use_cuda_graph,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables,
            seq_kv_lens=seq_kv_lens,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            # cloned_block_tables=cloned_block_tables,
            # seq_kv_lens_before_prune=seq_kv_lens_before_prune
            )

        # Batch may be composed of prefill|decodes, adjust query start indices
        # to refer to the start of decodes when the two are split apart.
        # E.g. in tokens:[3 prefills|6 decodes], query_start_loc=[3,9] => [0,6].
        if self._cached_decode_metadata.query_start_loc is not None:
            qs = self._cached_decode_metadata.query_start_loc
            self._cached_decode_metadata.query_start_loc = qs - qs[0]
        return self._cached_decode_metadata

    def get_seq_lens(
        self,
        attn_type: str,
    ):
        '''
        Extract appropriate sequence lengths from attention metadata
        according to attention type.

        Arguments:

        * attn_metadata: Attention metadata structure associated with attention
        * attn_type: encoder attention, decoder self-attention,
                    encoder/decoder cross-attention

        Returns:
        * Appropriate sequence lengths tensor for query
        * Appropriate sequence lengths tensor for key & value
        '''
        if (attn_type == AttentionType.DECODER
                or attn_type == AttentionType.ENCODER_ONLY):
            seq_lens_q = self.seq_lens
            if self.seq_kv_lens is None:
                seq_lens_kv = self.seq_lens
            else:
                seq_lens_kv = self.seq_kv_lens
        elif attn_type == AttentionType.ENCODER:
            seq_lens_q = self.encoder_seq_lens
            seq_lens_kv = self.encoder_seq_lens
        elif attn_type == AttentionType.ENCODER_DECODER:
            seq_lens_q = self.seq_lens
            seq_lens_kv = self.encoder_seq_lens
        else:
            raise AttributeError(f"Invalid attention type {str(attn_type)}")
        return seq_lens_q, seq_lens_kv

    def get_attn_bias(
        self,
        attn_type: str,
    ) -> Optional[List[torch.Tensor]]:
        '''
        Extract appropriate attention bias from attention metadata
        according to attention type.

        Arguments:

        * attn_metadata: Attention metadata structure associated with attention
        * attn_type: encoder attention, decoder self-attention,
                    encoder/decoder cross-attention

        Returns:
        * Appropriate attention bias value given the attention type
        '''

        if (attn_type == AttentionType.DECODER
                or attn_type == AttentionType.ENCODER_ONLY):
            return self.attn_bias
        elif attn_type == AttentionType.ENCODER:
            return self.encoder_attn_bias
        elif attn_type == AttentionType.ENCODER_DECODER:
            return self.cross_attn_bias
        else:
            raise AttributeError(f"Invalid attention type {str(attn_type)}")

    def set_attn_bias(
        self,
        attn_bias: List[torch.Tensor],
        attn_type: str,
    ) -> None:
        '''
        Update appropriate attention bias field of attention metadata,
        according to attention type.

        Arguments:

        * attn_metadata: Attention metadata structure associated with attention
        * attn_bias: The desired attention bias value
        * attn_type: encoder attention, decoder self-attention,
                    encoder/decoder cross-attention
        '''

        if (attn_type == AttentionType.DECODER
                or attn_type == AttentionType.ENCODER_ONLY):
            self.attn_bias = attn_bias
        elif attn_type == AttentionType.ENCODER:
            self.encoder_attn_bias = attn_bias
        elif attn_type == AttentionType.ENCODER_DECODER:
            self.cross_attn_bias = attn_bias
        else:
            raise AttributeError(f"Invalid attention type {str(attn_type)}")

    def get_seq_len_block_table_args(
        self,
        attn_type: str,
    ) -> tuple:
        '''
        The particular choice of sequence-length- and block-table-related
        attributes which should be extracted from attn_metadata is dependent
        on the type of attention operation.

        Decoder attn -> select entirely decoder self-attention-related fields
        Encoder/decoder cross-attn -> select encoder sequence lengths &
                                    cross-attn block-tables fields
        Encoder attn -> select encoder sequence lengths fields & no block tables

        Arguments:

        * attn_metadata: Attention metadata structure associated with attention
        * is_prompt: True if prefill, False otherwise
        * attn_type: encoder attention, decoder self-attention,
                    encoder/decoder cross-attention

        Returns:

        * Appropriate sequence-lengths tensor
        * Appropriate max sequence-length scalar
        * Appropriate block tables (or None)
        '''
        if (attn_type == AttentionType.DECODER
                or attn_type == AttentionType.ENCODER_ONLY):
            # Decoder self-attention
            # Choose max_seq_len based on whether we are in prompt_run
            return (self.seq_lens_tensor, self.max_decode_seq_len,
                        self.block_tables)
        elif attn_type == AttentionType.ENCODER_DECODER:
            # Enc/dec cross-attention KVs match encoder sequence length;
            # cross-attention utilizes special "cross" block tables
            return (self.encoder_seq_lens_tensor, self.max_encoder_seq_len,
                    self.cross_block_tables)
        elif attn_type == AttentionType.ENCODER:
            # No block tables associated with encoder attention
            return (self.encoder_seq_lens_tensor, self.max_encoder_seq_len,
                    None)
        else:
            raise AttributeError(f"Invalid attention type {str(attn_type)}")
        
####################################################################################################  
class TorchCUDASDPAMetadataBuilder(
        AttentionMetadataBuilder[TorchCUDASDPAMetadata]):
    
    def __init__(self, input_builder: "ModelInputForGPUBuilder"):
        self.slot_mapping: List[int] = []
        self.prefill_seq_lens: List[int] = []
        self.context_lens: List[int] = []
        self.block_tables: List[List[int]] = []
        self.curr_seq_lens: List[int] = []
        self.multimodal_placeholder_maps: Dict[
            str,
            MultiModalPlaceholderMap] = defaultdict(MultiModalPlaceholderMap)
        self.num_prefills = 0
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0

        self.input_builder = input_builder
        self.runner = input_builder.runner

        self.sliding_window = input_builder.sliding_window
        self.block_size = input_builder.block_size
        self.paged_evict_config = self.runner.cache_config.paged_evict_config
        self.seq_kv_lens: List[int] = [] # only used when paged_evict is enabled
        # self.seq_kv_lens_before_prune: List[int] = [] # only used when paged_evict is enabled
        # self.target_block_tables: List[List[int]] = []
        self.reqid_mapping_blocktables: Dict[str, Dict[int, List[int]]] = {} # save the blocktables for each request group 
        
    def _add_seq_group(
            self, inter_data: "ModelInputForGPUBuilder.InterDataForSeqGroup",
            chunked_prefill_enabled: bool):
        
        is_prompt = inter_data.is_prompt
        block_tables = inter_data.block_tables
        
        self.reqid_mapping_blocktables[inter_data.request_id] = inter_data.block_tables
        
        for (seq_id, token_len, seq_len, curr_seq_len, query_len, context_len,
             curr_sliding_window_block) in zip(
                 inter_data.seq_ids, [len(t) for t in inter_data.input_tokens],
                 inter_data.orig_seq_lens, inter_data.seq_lens,
                 inter_data.query_lens, inter_data.context_lens,
                 inter_data.curr_sliding_window_blocks):
            # print(f"TorchCUDASDPAMetdataBuilder: seq_id: {seq_id}, token_len: {token_len}, seq_len: {seq_len}, "
            #       f"curr_seq_len: {curr_seq_len}, query_len: {query_len}, context_len: {context_len}, curr_sliding_window_block: {curr_sliding_window_block}"
            #       f", is_prompt: {is_prompt}, block_tables: {block_tables}")
            self.context_lens.append(context_len)
            if is_prompt:
                mm_maps = inter_data.multi_modal_placeholder_maps
                if mm_maps:
                    for modality, placeholders in mm_maps.items():
                        self.multimodal_placeholder_maps[modality].extend(
                            placeholders)

                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                assert query_len == 1, (
                    "seq_len: {}, context_len: {}, query_len: {}".format(
                        seq_len, context_len, query_len))
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

            # Compute block table.
            # TODO(sang): Combine chunked prefill and prefix caching by
            # only allowing multiple of block_size chunk size.
            # NOTE: This only works for oooooooxxx style attention.
            block_table = []
            if inter_data.prefix_cache_hit:
                block_table = block_tables[seq_id]
            elif ((chunked_prefill_enabled or not is_prompt)
                  and block_tables is not None):
                if curr_sliding_window_block == 0:
                    block_table = block_tables[seq_id]
                else:
                    block_table = block_tables[seq_id][
                        -curr_sliding_window_block:]
            self.block_tables.append(block_table)

            # Compute slot mapping.
            is_profile_run = is_block_tables_empty(block_tables)
            start_idx = compute_slot_mapping_start_idx(is_prompt, query_len,
                                                       context_len,
                                                       self.sliding_window)
            
            if self.paged_evict_config is None:
                compute_slot_mapping(is_profile_run, self.slot_mapping, seq_id,
                                    seq_len, context_len, start_idx,
                                    self.block_size, inter_data.block_tables)
            else:
                #### TODO: modified this part for eviction method
                # compute prunned block_table
                # if not is_profile_run:
                #     block_table_tmp = inter_data.target_block_tables[seq_id]
                #     self.target_block_tables.append(block_table_tmp)
                
                self._compute_slot_mapping(is_profile_run, is_prompt, self.slot_mapping, 
                                           seq_id, query_len, context_len, self.block_size, inter_data.block_tables)
            
            
        if inter_data.seq_kv_lens is not None:
            self.seq_kv_lens.extend(inter_data.seq_kv_lens.values())
            # self.seq_kv_lens_before_prune.extend(inter_data.seq_kv_lens_before_prune.values())
            
    def _compute_slot_mapping(self, is_profile_run: bool, is_prompt: bool, slot_mapping: List[int],
                              seq_id: int, query_len:int, context_len: int, block_size:int, block_tables:List[List[int]]):
        assert self.sliding_window is None, "Sliding window is not supported when paged_evict is enabled"
        if is_profile_run:
            # TODO: This line needs to change
            slot_mapping.extend([PAD_SLOT_ID] * query_len)
            return
        
        # get the block_table for the current sequence
        # TODO: optimize the following code to improve performance.
        ### TODO: Update for calculating the slot_mapping
        block_table = block_tables[seq_id]
        if is_prompt:
            remainder = query_len % block_size
            if remainder == 0:
                for block_id in block_table:
                    block_start_offset = block_id * block_size
                    slot_mapping.extend([ block_start_offset + i for i in range(0, block_size)])
            else:
                for block_id in block_table[:-1]:
                    block_start_offset = block_id * block_size
                    slot_mapping.extend([ block_start_offset + i for i in range(0, block_size)])
                
                block_start_offset = block_table[-1] * block_size
                slot_mapping.extend([ block_start_offset + i for i in range(0, remainder)])
                
        else:
            # decode request
            assert query_len == 1, "decode request should have query_len=1 when paged_evict is enabled"
            # get the start index of the slot_mapping (i.e, seen tokens in the last blocks)
            start_idx = context_len % block_size
            end_idx = start_idx + query_len
            block_start_offset = block_table[-1] * block_size 
            slot_mapping.extend([ block_start_offset + i for i in range(start_idx, end_idx)])

        
    def build(self, seq_lens: List[int], query_lens: List[int],
              cuda_graph_pad_size: int, batch_size: int):
        """Build attention metadata with on-device tensors.

        Args:
            seq_lens: The maybe padded sequence lengths of the input sequences.
            query_lens: The query lengths of the input sequences.
            cuda_graph_pad_size: The padding size for cuda graph.
                                 -1 if cuda graph is not used.
            batch_size: The maybe padded batch size.
        """
        for inter_data in self.input_builder.inter_data_list:
            self._add_seq_group(inter_data,
                                self.input_builder.chunked_prefill_enabled)

        device = self.runner.device
        use_captured_graph = cuda_graph_pad_size != -1

        max_query_len = max(query_lens)
        max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
        max_decode_seq_len = max(self.curr_seq_lens, default=0)
        num_decode_tokens = self.num_decode_tokens
        query_start_loc = list(accumulate(query_lens, initial=0))
        seq_start_loc = list(accumulate(seq_lens, initial=0))

        if use_captured_graph:
            self.slot_mapping.extend([PAD_SLOT_ID] * cuda_graph_pad_size)
            self.block_tables.extend([] * cuda_graph_pad_size)
            num_decode_tokens = batch_size

            # The shape of graph_block_tables is
            # [max batch size, max context len // block size].
            input_block_tables = self.runner.graph_block_tables[:batch_size]
            for i, block_table in enumerate(self.block_tables):
                if block_table:
                    input_block_tables[i, :len(block_table)] = block_table
            block_tables = torch.from_numpy(input_block_tables).to(
                device, non_blocking=True)
            # This one needs to change, It is not correct
            # target_block_tables = block_tables.clone()
        else:
            block_tables = make_tensor_with_pad(
                self.block_tables,
                pad=0,
                dtype=torch.int,
                device=device,
            )
            # target_block_tables = make_tensor_with_pad(
            #     self.target_block_tables,
            #     pad=0,
            #     dtype=torch.int,
            #     device=device,
            # )
        assert max_query_len > 0, "query_lens: {}".format(query_lens)

        assert device is not None
        context_lens_tensor = async_tensor_h2d(self.context_lens, torch.int,
                                               device, self.runner.pin_memory)
        seq_lens_tensor = async_tensor_h2d(seq_lens, torch.int, device,
                                           self.runner.pin_memory)
        seq_kv_lens_tensor = async_tensor_h2d(self.seq_kv_lens, torch.int, device,
                                            self.runner.pin_memory)
        slot_mapping_tensor = async_tensor_h2d(self.slot_mapping, torch.long,
                                               device, self.runner.pin_memory)
        query_start_loc_tensor = async_tensor_h2d(query_start_loc, torch.int32,
                                                  device,
                                                  self.runner.pin_memory)
        seq_start_loc_tensor = async_tensor_h2d(seq_start_loc, torch.int32,
                                                device, self.runner.pin_memory)
        placeholder_index_maps = {
            modality: placeholder_map.index_map()
            for modality, placeholder_map in
            self.multimodal_placeholder_maps.items()
        }

        return TorchCUDASDPAMetadata(  # type: ignore
            num_prefills=self.num_prefills,
            slot_mapping=slot_mapping_tensor,
            multi_modal_placeholder_index_maps=placeholder_index_maps,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=max_query_len,
            max_prefill_seq_len=max_prefill_seq_len,
            max_decode_seq_len=max_decode_seq_len,
            query_start_loc=query_start_loc_tensor,
            seq_start_loc=seq_start_loc_tensor,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=use_captured_graph,
            seq_kv_lens=self.seq_kv_lens,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            # cloned_block_tables=target_block_tables,
            # seq_kv_lens_before_prune=self.seq_kv_lens_before_prune,
            reqid_mapping_blocktables=self.reqid_mapping_blocktables
        )

###################################################################################################
class CudaStreamManager:
    """Manage a fixed size of streams for each device"""
    _instances: Dict[str, "CudaStreamManager"] = {}  # Stores manager instances per device
    
    def __init__(self, device_id: str, num_streams: int):
        self.device_id = device_id
        self.num_streams = num_streams
        self.streams = [torch.cuda.Stream(device=device_id) for _ in range(num_streams)]
    
    @staticmethod
    def get_instance(device_id: str) -> "CudaStreamManager":
        """
        Returns the singleton instance of CudaStreamManager for the given device.
        Creates it if it does not exist.
        """
        if device_id not in CudaStreamManager._instances:
            num_streams = 8
            CudaStreamManager._instances[device_id] = CudaStreamManager(device_id, num_streams)
        return CudaStreamManager._instances[device_id]
    
    def get_streams(self) -> List[torch.cuda.Stream]:
        """
        Returns the list of CUDA streams for the given device.
        Ensures that streams are created only once.
        """
        return self.streams
    
####################################################################################################
class TorchCUDASDPABackendImpl(AttentionImpl[TorchCUDASDPAMetadata]):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None, 
        **kwargs
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "Torch SPDA does not support block-sparse attention.")
        if logits_soft_cap is not None:
            print_warning_once("Torch SPDA does not support logits soft cap. "
                               "Outputs may be slightly off.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.need_mask = (self.alibi_slopes is not None
                          or self.sliding_window is not None)

        supported_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {supported_head_sizes}.")
        if kv_cache_dtype != "auto":
            raise NotImplementedError(
                "Torch SDPA backend does not support FP8 KV cache. "
                "Please use xFormers backend instead.")
        
        self.kv_cache_pruner = None
        self.cache_config:CacheConfig = kwargs.get('cache_config', None)
        self.enabled_paged_eviction = False
        self.streams = None
        if ((self.cache_config is not None) and (self.cache_config.paged_evict_config is not None)):
            self.enabled_paged_eviction = True
            self.paged_evict_config = self.cache_config.paged_evict_config 
            # print(f"Enable/Disable evict prefill: {self.paged_evict_config.disable_evict_prefill}")
            self.kv_cache_pruner = KVCachePruner(
                                self.paged_evict_config.cache_prune_type,
                                self.paged_evict_config.evict_method,
                                self.cache_config.block_size, 
                                self.paged_evict_config.cache_budget,           
                                self.paged_evict_config.initial_blocks
                                ) 
            self.block_l2norm_mgr = BlockL2NormManager()
            self.stream_manager = CudaStreamManager.get_instance(f"cuda:{torch.cuda.current_device()}")
            self.streams = self.stream_manager.get_streams()

        self.prefix: str = kwargs.get('prefix', "")
        self.layer_idx: int = extract_layer_index(self.prefix) if self.prefix != "" else -1
        
        random.seed(1234)
        # record the blockId to be evicted for each request 
        self.reqids_mapping_rmv_blockIdx = {}
        
        # just for test case
        # for local and global, if enable_random_evict is enabled, it will randomly evict a block instead of compare the l2_norms.
        self.enable_random_evict = True

        # max_num_seqs = 256 # TODO: change it to use the configuration parameter from scheduleConfig
        # self.cached_keys = torch.zeros((num_update_tokens, self.num_kv_heads, self.head_size),
        #                                 dtype=kv_cache_dtype, device='cuda')
        # self.cached_values = torch.zeros((num_update_tokens, self.num_kv_heads, self.head_size), 
        #                                   dtype=kv_cache_dtype, device='cuda')
        self.tmp_cache_singleton = TmpCacheSingleton.get_instance()
        assert self.tmp_cache_singleton is not None, "TmpCacheSingleton should be initialized before TorchCUDASDPABackendImpl"

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TorchCUDASDPAMetadata,  # type: ignore
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: str = AttentionType.DECODER,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with torch SDPA and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
                NOTE: kv_cache will be an empty tensor with shape [0]
                for profiling run. 
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert k_scale == 1.0 and v_scale == 1.0
        if (attn_type == AttentionType.ENCODER
                and (not attn_metadata.is_all_encoder_attn_metadata_set)):
            raise AttributeError("Encoder attention requires setting "
                                 "encoder metadata attributes.")
        elif (attn_type == AttentionType.ENCODER_DECODER
              and (not attn_metadata.is_all_cross_attn_metadata_set)):
            raise AttributeError("Encoder/decoder cross-attention "
                                 "requires setting cross-attention "
                                 "metadata attributes.")

        # Reshape the query, key, and value tensors. Shape: [num_tokens, num_heads, head_size]
        query = query.view(-1, self.num_heads, self.head_size)
        if key is not None:
            assert value is not None
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
        else:
            assert value is None
        
        is_prune_decode, decode_block_tables = False, None
        is_profile_run = True if kv_cache.numel() == 0 else False
        # Scenario where KV cache is non-empty tensor (here the KV cache is the original KV cache blocks)
        if self.enabled_paged_eviction == False:
            if (attn_type != AttentionType.ENCODER and kv_cache.numel() > 0):
                key_cache, value_cache = self._update_kvcache(attn_type, key, value, kv_cache, attn_metadata.slot_mapping, attn_metadata.cross_slot_mapping, k_scale, v_scale)
        else: 
            assert attn_type == AttentionType.DECODER, "Paged eviction is only enabled for decoder attention"
            # Paged eviction is enabled case
            if kv_cache.numel() == 0 and self.paged_evict_config.disable_evict_prefill == False:
                # For profiling, we need to prunne the input key and value.
                start = time.perf_counter()
                # key, value = self._prune_prompts_keys_and_values(key, value, attn_metadata, profile_run=True)
                key, value = self._prune_prompts_keys_and_values_new(key, value, attn_metadata, profile_run=True)
                prune_time = time.perf_counter() - start
                logger.debug(f"Prune {len(attn_metadata.seq_lens)} requests during profiling cost {prune_time:.6f} seconds")
            
            if (attn_type != AttentionType.ENCODER and kv_cache.numel() > 0):
                # prompt requests or decode requests
                assert attn_metadata.prefill_metadata is None or attn_metadata.decode_metadata is None, "paged eviction does not support chunked prefill"
                ### update the batched requests
                t1 = time.perf_counter()
                self.block_l2norm_mgr.update_reqs(attn_metadata.reqid_mapping_blocktables)
                t2 = time.perf_counter() - t1
                if attn_metadata.prefill_metadata is not None:
                    # prefill requests: 1) prune the input, key and value; 2) write the prunned key and value to kv_cache
                    # Note the num_tokens if key[0] is the sum of all requests, so we should processed requests one by one
                    if self.paged_evict_config.disable_evict_prefill == False:
                        start = time.perf_counter()
                        # key, value = self._prune_prompts_keys_and_values(key, value, attn_metadata, profile_run=False)
                        key, value = self._prune_prompts_keys_and_values_new(key, value, attn_metadata, profile_run=False)
                        prune_time = time.perf_counter() - start
                        logger.debug(f"With prefill-prune case: prune {len(attn_metadata.seq_lens)} prompt requests cost {prune_time:.6f} seconds. len(key) = {len(key)}, len(value) = {len(value)}, update_tables cost {t2:.6f} seconds")
                        # write the prunned key and value to kv_cache: the slotting_map is already the prunned one
                        key_cache, value_cache = self._update_kvcache(attn_type, key, value, kv_cache, attn_metadata.slot_mapping, attn_metadata.cross_slot_mapping, k_scale, v_scale)
                    else:
                        # for this case, we will perform the prunning after attention calculation
                        pass
                else:
                    if self.paged_evict_config.evict_method in ['streamingLLM', 'streamingLLM-1']:
                        ## Processe streamingLLM method
                        start = time.perf_counter()
                        # prunned_keys, prunned_values, prunned_kv_slot_mapping, is_pruned_decode, \
                        #     updated_block_tables, seqid_to_rmv_block_idx = \
                        #     self._prune_decode_reqs_streamingLLM(kv_cache, attn_metadata)
                        num_update_tokens, pruned_kvs_slot_mapping, is_pruned_decode, updated_block_tables, seqid_to_rmv_block_idx = \
                            self._prune_decode_reqs_streamingLLM(kv_cache, attn_metadata)
                        # prunned_keys, prunned_values, prunned_kvs_slot_mapping = None,
                        prune_time = time.perf_counter() - start
                        logger.debug(f"Prune decode requests cost {prune_time:.6f} seconds. is_pruned_decode={is_pruned_decode} num_update_tokens={num_update_tokens} ")
                        
                        self.reqids_mapping_rmv_blockIdx = seqid_to_rmv_block_idx 
                        if is_pruned_decode:
                            decode_block_tables = updated_block_tables
                        
                        # if prunned_keys is not None:
                        #     # need to merge the prunned keys with key, prunned values with value, and slot_mapping with mapping
                        #     prunned_keys = torch.cat([key, prunned_keys], dim=0)
                        #     prunned_values = torch.cat([value, prunned_values], dim=0)
                        #     prunned_kv_slot_mapping = torch.cat([attn_metadata.slot_mapping, prunned_kv_slot_mapping], dim=0)
                        #     key_cache, value_cache = self._update_kvcache(attn_type, prunned_keys, prunned_values, kv_cache, prunned_kv_slot_mapping, attn_metadata.cross_slot_mapping, k_scale, v_scale)
                        if num_update_tokens > 0:
                            total_keys = key.shape[0] + num_update_tokens
                            total_values = value.shape[0] + num_update_tokens
                            pruned_keys = self.tmp_cache_singleton.get_cached_keys(total_keys)
                            pruned_values = self.tmp_cache_singleton.get_cached_values(total_values)
                            # update the cached keys and values
                            pruned_keys[:key.shape[0]] = key
                            pruned_values[:value.shape[0]] = value
                            # assign 0 to the rest of the cached keys and values
                            pruned_keys[key.shape[0]:] = 0
                            pruned_values[value.shape[0]:] = 0
                            
                            pruned_kvs_slot_mapping = torch.cat([attn_metadata.slot_mapping, pruned_kvs_slot_mapping], dim=0)
                            
                            key_cache, value_cache = self._update_kvcache(attn_type, pruned_keys, pruned_values, kv_cache, pruned_kvs_slot_mapping, attn_metadata.cross_slot_mapping, k_scale, v_scale)
                        else:
                            # no prunning, just update the input key and value to the kv_cache
                            key_cache, value_cache = self._update_kvcache(attn_type, key, value, kv_cache, attn_metadata.slot_mapping, attn_metadata.cross_slot_mapping, k_scale, v_scale)
                            
                    elif self.paged_evict_config.evict_method in ['local', 'global']:
                        # local or global comparison
                        # update the block_tables if need to prune data
                        start = time.perf_counter()
                        decode_block_tables, seqid_to_rmv_block_idx, is_prune_decode = \
                            self._prune_decode_reqs_topk(kv_cache, attn_metadata)
                        prune_time = time.perf_counter() - start
                        logger.debug(f"Prune decode requests cost {prune_time:.6f} seconds. is_prune_decode={is_prune_decode}, decode_block_tables={decode_block_tables}")
                        self.reqids_mapping_rmv_blockIdx = seqid_to_rmv_block_idx
                        key_cache, value_cache = self._update_kvcache(attn_type, key, value, kv_cache, attn_metadata.slot_mapping, attn_metadata.cross_slot_mapping, k_scale, v_scale)
                        
                    elif self.paged_evict_config.evict_method in ['inverse_key_l2']:
                        ## Processe inverse_key_l2 method
                        start = time.perf_counter()
                        # prunned_keys, prunned_values, prunned_kvs_slot_mapping, is_pruned_decode = self._prune_decode_reqs_inverseKeyL2(kv_cache, attn_metadata)
                        num_update_tokens, prunned_kvs_slot_mapping, is_pruned_decode = self._prune_decode_reqs_inverseKeyL2(kv_cache, attn_metadata)
                        prune_time = time.perf_counter() - start
                        logger.debug(f"Prune decode requests cost {prune_time:.6f} seconds. is_pruned_decode={is_pruned_decode}")
                        # if prunned_keys is not None:
                        #     # need to merge the prunned keys with key, prunned values with value, and slot_mapping with mapping
                        #     prunned_keys = torch.cat([key, prunned_keys], dim=0)
                        #     prunned_values = torch.cat([value, prunned_values], dim=0)
                        #     prunned_kvs_slot_mapping = torch.cat([attn_metadata.slot_mapping, prunned_kvs_slot_mapping], dim=0)
                        #     key_cache, value_cache = self._update_kvcache(attn_type, prunned_keys, prunned_values, kv_cache, prunned_kvs_slot_mapping, attn_metadata.cross_slot_mapping, k_scale, v_scale)
                        if num_update_tokens > 0:
                            total_keys = key.shape[0] + num_update_tokens
                            total_values = value.shape[0] + num_update_tokens
                            pruned_keys = self.tmp_cache_singleton.get_cached_keys(total_keys)
                            pruned_values = self.tmp_cache_singleton.get_cached_values(total_values)
                            # update the cached keys and values
                            pruned_keys[:key.shape[0]] = key
                            pruned_values[:value.shape[0]] = value
                            # assign 0 to the rest of the cached keys and values
                            pruned_keys[key.shape[0]:] = 0
                            pruned_values[value.shape[0]:] = 0
                            
                            prunned_kvs_slot_mapping = torch.cat([attn_metadata.slot_mapping, prunned_kvs_slot_mapping], dim=0)

                            key_cache, value_cache = self._update_kvcache(attn_type, pruned_keys, pruned_values, kv_cache, prunned_kvs_slot_mapping, attn_metadata.cross_slot_mapping, k_scale, v_scale)
                        else:
                            key_cache, value_cache = self._update_kvcache(attn_type, key, value, kv_cache, attn_metadata.slot_mapping, attn_metadata.cross_slot_mapping, k_scale, v_scale) 
                    else:
                        raise NotImplementedError(f"Unknown eviction method: {self.paged_evict_config.evict_method}")

        if attn_type != AttentionType.ENCODER:
            # Decoder self-attention supports chunked prefill.
            # Encoder/decoder cross-attention requires no chunked
            # prefill (100% prefill or 100% decode tokens, no mix)
            num_prefill_tokens = attn_metadata.num_prefill_tokens
            num_decode_tokens = attn_metadata.num_decode_tokens
        else:
            # Encoder attention - chunked prefill is not applicable;
            # derive token-count from query shape & and treat them
            # as 100% prefill tokens
            assert attn_metadata.num_encoder_tokens is not None
            num_prefill_tokens = attn_metadata.num_encoder_tokens
            num_decode_tokens = 0

        # if attn_type == AttentionType.DECODER:
        #     # Only enforce this shape-constraint for decoder
        #     # self-attention
        #     assert key.shape[0] == num_prefill_tokens + num_decode_tokens
        #     assert value.shape[0] == num_prefill_tokens + num_decode_tokens

        output = torch.empty_like(query)
        if prefill_meta := attn_metadata.prefill_metadata:
            assert attn_metadata.seq_lens is not None
            if (kv_cache.numel() == 0 or prefill_meta.block_tables is None
                    or prefill_meta.block_tables.numel() == 0):
                # prefill case or profiling run case
                self._run_sdpa_forward(output,
                                       query,
                                       key,
                                       value,
                                       prefill_meta,
                                       attn_type=attn_type)
                # When paged_eviction is enabled and disable evict prefill, we will write the KV cache here.
                if self.enabled_paged_eviction and self.paged_evict_config.disable_evict_prefill:
                    # Only prune the prefill tokens when it not the profile run.
                    if is_profile_run == False:
                        start = time.perf_counter()
                        # key, value = self._prune_prompts_keys_and_values(key, value, attn_metadata, profile_run=False)
                        key, value = self._prune_prompts_keys_and_values_new(key, value, attn_metadata, profile_run=False)
                        prune_time = time.perf_counter() - start
                        logger.debug(f"Without prefill-prune case: prune {len(attn_metadata.seq_lens)} prompt requests cost {prune_time:.6f} seconds. len(key) = {len(key)}, len(value) = {len(value)}, update_tables cost {t2:.6f} seconds")
                        # write the prunned key and value to kv_cache: the slotting_map is already the prunned one
                        key_cache, value_cache = self._update_kvcache(attn_type, key, value, kv_cache, attn_metadata.slot_mapping, attn_metadata.cross_slot_mapping, k_scale, v_scale) 
                
            else:
                # We disabled the chunked-prefill for now, so this part will not be executed
                logger.error(f"ERROR: Go here. the request may be use chunked prefill, which is not supported when eviction is enabled.")
                raise NotImplementedError(
                    "Chunked prefill is not supported when paged eviction is enabled.")

        if decode_meta := attn_metadata.decode_metadata:
            assert attn_type != AttentionType.ENCODER_ONLY, (
                "Encoder-only models should not have decode metadata.")
            # Decoding run.
            # TODO: Confirm that for paged eviction enabled case, the seq_lens_args shoud be 
            # seq_kv_lens and max_seq_len_args should max_seq_len_args
            (
                seq_lens_arg,
                max_seq_len_arg,
                block_tables_arg,
            ) = decode_meta.get_seq_len_block_table_args(attn_type)
            
            if self.enabled_paged_eviction:
                seq_lens_arg = decode_meta.seq_kv_lens_tensor
                if is_prune_decode and decode_block_tables is not None:
                    block_tables_arg = decode_block_tables
            
            output[attn_metadata.num_prefill_tokens:] = PagedAttention.forward_decode(
                query[attn_metadata.num_prefill_tokens:],
                key_cache,
                value_cache,
                block_tables_arg,
                seq_lens_arg,
                max_seq_len_arg,
                self.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
                k_scale,
                v_scale,
            )
        # Reshape the output tensor.
        return output.view(-1, self.num_heads * self.head_size)

    def get_rmv_block_idx(self):
        return self.reqids_mapping_rmv_blockIdx

    def _run_sdpa_forward(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: TorchCUDASDPAMetadata,
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        if self.num_kv_heads != self.num_heads:
            key = key.repeat_interleave(self.num_queries_per_kv, dim=1)
            value = value.repeat_interleave(self.num_queries_per_kv, dim=1)
        
        # This one needs to change
        attn_masks = attn_metadata.get_attn_bias(attn_type)
        if attn_masks is None:
            # print(f"In TorchCUDASDPABackendImpl._run_sdpa_forward: attn_masks is None")
            if self.alibi_slopes is not None:
                attn_masks = _make_alibi_bias(
                    self.alibi_slopes, query.dtype,
                    attn_metadata.seq_lens)  # type: ignore
            elif self.sliding_window is not None:
                assert attn_metadata.seq_lens is not None
                attn_masks = _make_sliding_window_bias(
                    attn_metadata.seq_lens, self.sliding_window,
                    query.dtype)  # type: ignore
            else:
                seq_lens, _ = attn_metadata.get_seq_lens(attn_type)
                attn_masks = [None] * len(seq_lens)
            attn_metadata.set_attn_bias(attn_masks, attn_type)

        # Rearrange the query, key, and value tensors. 
        # Shape change from [num_tokens, num_heads, head_size] to [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)
        key = key.movedim(0, key.dim() - 2)
        value = value.movedim(0, value.dim() - 2)

        causal_attn = (attn_type == AttentionType.DECODER)

        seq_lens_q, seq_lens_kv = attn_metadata.get_seq_lens(attn_type)
        if self.enabled_paged_eviction and self.paged_evict_config.disable_evict_prefill:
            # set seq_lens_kv equal to seq_lens_q
            seq_lens_kv = seq_lens_q
        
        start_q, start_kv = 0, 0
        for seq_len_q, seq_len_kv, mask in zip(seq_lens_q, seq_lens_kv,
                                               attn_masks):
            end_q = start_q + seq_len_q
            end_kv = start_kv + seq_len_kv
            sub_out = scaled_dot_product_attention(
                query[None, :, start_q:end_q, :],
                key[None, :, start_kv:end_kv, :],
                value[None, :, start_kv:end_kv, :],
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=causal_attn and mask is None,
                scale=self.scale).squeeze(0).movedim(query.dim() - 2, 0)
            output[start_q:end_q, :, :] = sub_out
            start_q, start_kv = end_q, end_kv
            
    def _update_kvcache(
            self, 
            attn_type: str, 
            key: torch.Tensor, 
            value: torch.Tensor, 
            kv_cache: torch.Tensor, 
            slot_mapping: torch.Tensor,
            cross_slot_mapping: torch.Tensor,
            k_scale: float, 
            v_scale: float,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        # KV-cache during decoder-self- or
        # encoder-decoder-cross-attention, but not
        # during encoder attention.
        #
        # Even if there are no new key/value pairs to cache,
        # we still need to break out key_cache and value_cache
        # i.e. for later use by paged attention
        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache, self.num_kv_heads, self.head_size)

        if (key is not None) and (value is not None):

            if attn_type == AttentionType.ENCODER_DECODER:
                # Update cross-attention KV cache (prefill-only)
                # During cross-attention decode, key & value will be None,
                # preventing this IF-statement branch from running
                updated_slot_mapping = cross_slot_mapping
            else:
                # Update self-attention KV cache (prefill/decode)
                updated_slot_mapping = slot_mapping

            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory
            # profiling run.
            PagedAttention.write_to_paged_cache(key, value, key_cache,
                                                value_cache,
                                                updated_slot_mapping,
                                                self.kv_cache_dtype,
                                                k_scale, v_scale)
        
        return key_cache, value_cache 
    
    # Parallel Implementation --- Version 2
    def _prune_prompts_keys_and_values(self, key: torch.Tensor, value: torch.Tensor, attn_metadata: TorchCUDASDPAMetadata, profile_run: bool):
        # start = time.perf_counter()
        num_streams = len(self.streams)
        num_requests = len(attn_metadata.seq_lens)
        pruned_keys, pruned_values = [None] * num_requests, [None] * num_requests  # Pre-allocate lists
        
        s_idx, e_idx = 0, 0
        # Assign requests to streams in a round-robin fashion and run the given number of streams in parallel
        for i, seq_len in enumerate(attn_metadata.seq_lens):
            e_idx += seq_len
            stream=self.streams[i % num_streams]  # Distribute across limited streams
            with torch.cuda.stream(stream):  # Switch to a separate CUDA stream
                ##### ToDo: The prune_prompt function need to return the l2 norm of each prunned block 
                pruned_keys[i], pruned_values[i] = self.kv_cache_pruner.prune_prompt(
                    key[s_idx:e_idx, :, :], value[s_idx:e_idx, :, :]
                )
                if not profile_run and self.paged_evict_config.evict_method in ["local", "global"]:
                    if self.enable_random_evict == False:
                        # For local and global prunning, we need to get the l2 norm of the prunned blocks if random evict is disabled.
                    # Note: the pruned_keys[i] and pruned_values[i] are already prunned, so we can get the l2 norm directly
                        l2_norms = []
                        for idx in range(pruned_keys[i].shape[0] // self.cache_config.block_size):
                            # Get the l2 norm of each prunned block
                            # Note: pruned_keys[i] and pruned_values[i] are already prunned, so we can get the l2 norm directly
                            blk_start = idx * self.cache_config.block_size
                            blk_end = (blk_start + 1) * self.cache_config.block_size
                            blk_keys = pruned_keys[i][blk_start:blk_end, :, :]
                            blk_values = pruned_values[i][blk_start:blk_end, :, :]
                            l2_norms.append(self.kv_cache_pruner.get_block_score(blk_keys, blk_values, self.paged_evict_config.evict_method))
                        # print(f"++++ req {i} has {pruned_keys[i].shape[0] // self.cache_config.block_size} blocks with l2 norms={l2_norms}")
                        self.block_l2norm_mgr.update_block_l2norms(i, l2_norms)
                
            s_idx = e_idx
        
        # end_1 = time.perf_counter()
        
        for stream in self.streams:
            torch.cuda.current_stream().wait_stream(stream)
            
        pruned_keys = torch.cat(pruned_keys, dim=0)
        pruned_values = torch.cat(pruned_values, dim=0)
         
        # end_2 = time.perf_counter()
        # logger.info(f"*********Prune {num_requests} prompts requests: end_1={end_1-start:.6f} end_2={end_2-start:.6f} on stream {torch.cuda.current_stream()}"
                    # f" prunned_keys.shape={pruned_keys.shape}, prunned_values.shape={pruned_values.shape}")

        return pruned_keys, pruned_values
    
    def _prune_prompts_keys_and_values_new(self, key: torch.Tensor, value: torch.Tensor, attn_metadata: TorchCUDASDPAMetadata, profile_run: bool):
        num_streams = len(self.streams)
        num_requests = len(attn_metadata.seq_lens)
        
        # Pre-calculate total output size to avoid concatenation
        is_prune_fill_reqs = False
        total_pruned_tokens = 0
        pruned_seq_lens = []
        for seq_len in attn_metadata.seq_lens:
            pruned_len = self.kv_cache_pruner.get_pruned_length(seq_len)  # You'll need this method
            if pruned_len < seq_len:
                is_prune_fill_reqs = True
            pruned_seq_lens.append(pruned_len)
            total_pruned_tokens += pruned_len
            
        if not is_prune_fill_reqs:
            return key, value  # No pruning needed, return original tensors
        
        # Pre-allocate final output tensors with exact size needed
        pruned_keys = torch.empty(
            (total_pruned_tokens, self.num_kv_heads, self.head_size),
            dtype=key.dtype,
            device=key.device
        )
        pruned_values = torch.empty(
            (total_pruned_tokens, self.num_kv_heads, self.head_size), 
            dtype=value.dtype,
            device=value.device
        )
        
        # Track output positions for direct writing
        output_start_positions = []
        current_pos = 0
        for pruned_len in pruned_seq_lens:
            output_start_positions.append(current_pos)
            current_pos += pruned_len
        
        s_idx, e_idx = 0, 0
        # Process requests in parallel, writing directly to final tensors
        for i, seq_len in enumerate(attn_metadata.seq_lens):
            e_idx += seq_len
            stream = self.streams[i % num_streams]
            
            # Get output slice positions
            out_start = output_start_positions[i]
            out_end = out_start + pruned_seq_lens[i]
            
            with torch.cuda.stream(stream):
                # Write directly to pre-allocated output tensors (no intermediate storage)
                self.kv_cache_pruner.prune_prompt_inplace(
                    key[s_idx:e_idx, :, :], 
                    value[s_idx:e_idx, :, :],
                    pruned_keys[out_start:out_end, :, :],  # Direct output write
                    pruned_values[out_start:out_end, :, :]  # Direct output write
                )
                
                if not profile_run and self.paged_evict_config.evict_method in ["local", "global"]:
                    if self.enable_random_evict == False:
                        l2_norms = []
                        for idx in range(pruned_seq_lens[i] // self.cache_config.block_size):
                            blk_start = out_start + idx * self.cache_config.block_size
                            blk_end = blk_start + self.cache_config.block_size
                            # Work directly with the final tensor slices
                            blk_keys = pruned_keys[blk_start:blk_end, :, :]
                            blk_values = pruned_values[blk_start:blk_end, :, :]
                            l2_norms.append(self.kv_cache_pruner.get_block_score(blk_keys, blk_values, self.paged_evict_config.evict_method))
                        self.block_l2norm_mgr.update_block_l2norms(i, l2_norms)
            
            s_idx = e_idx
        
        # Wait for all streams to complete
        for stream in self.streams:
            torch.cuda.current_stream().wait_stream(stream)
        
        return pruned_keys, pruned_values
           
    ### Parallel Implementation --- Version 2
    # def _prune_decode_reqs_oldest_block(self, kv_cache: torch.Tensor, attn_metadata: TorchCUDASDPAMetadata):
    #     # start_1 = time.perf_counter()
    #     kv_cache_device, kv_cache_dtype = kv_cache.device, kv_cache.dtype
    #     num_streams = len(self.streams) # use a fixed number of streams to avoid too many streams
    #     active_streams = {}  # Track streams that were actually used for pruning
        
    #     pruned_keys, pruned_values, prunned_kvs_slot_mapping = [], [], []
        
    #     is_pruned = False
    #     for id, seq_len in enumerate(attn_metadata.seq_lens):
    #         # Step-1: check if the request reach to the prunning point
    #         if seq_len % (self.cache_config.block_size * self.paged_evict_config.evict_freq) != 0:
    #             continue
            
    #         cur_seq_kv_len = attn_metadata.seq_kv_lens_before_prune[id]
    #         # Step-2: get the blocks to prune and merge for this request (for new, it is topk)
    #         s_idx_bt, e_idx_bt, prune_tokens = self.kv_cache_pruner.get_blocks_to_prune_and_merge_decode(cur_seq_kv_len)
    #         if s_idx_bt == -1:
    #             continue ## No pruning needed, skip further processing
            
    #         # This request needs pruning, so allocate a stream for it
    #         is_pruned = True 
    #         if self.cache_config.paged_evict_config.decode_evict_method == "streamingLLM":
    #             continue
    #         else:
    #             stream_id = id % num_streams  # Distribute only among needed streams
    #             if stream_id not in active_streams:
    #                 active_streams[stream_id] = self.streams[stream_id]  # Keep track of active streams
                
    #             # Other prune approach
    #             with torch.cuda.stream(self.streams[stream_id]):
    #                 # start = time.perf_counter()
    #                 num_tmp_blocks = e_idx_bt - s_idx_bt
    #                 block_size = self.cache_config.block_size
    #                 #### For local and global prunning, get the l2 norm of topk blocks
    #                 tmp_kvs_shape = PagedAttention.get_kv_cache_shape(num_tmp_blocks, 
    #                                         block_size, self.num_kv_heads, self.head_size)
    #                 tmp_kvs = torch.empty(tmp_kvs_shape, dtype=kv_cache_dtype, device=kv_cache_device)
    #                 # copy the corresponding blocks to the temp memory
    #                 src_indices = attn_metadata.decode_metadata.block_tables[id][s_idx_bt: s_idx_bt + num_tmp_blocks]
    #                 tmp_kvs[:, :num_tmp_blocks, :] = kv_cache[:, src_indices, :]
    #                 # then prune the tmp_kv_cache
    #                 tmp_key = tmp_kvs[0].view(-1, self.num_kv_heads, self.head_size)
    #                 tmp_value = tmp_kvs[1].view(-1, self.num_kv_heads, self.head_size)
    #                 # end_1 = time.perf_counter()
    #                 # print(f"+++++++TorchCUDASDPABackendImpl---2. tmp_key.shape={tmp_key.shape}, tmp_value.shape={tmp_value.shape}")
    #                 tmp_key, tmp_value=self.kv_cache_pruner.prune_oldest_block(tmp_key, tmp_value, prune_tokens)
    #                 # end_2 = time.perf_counter()
    #                 # print(f"+++++++TorchCUDASDPABackendImpl---3. tmp_key.shape={tmp_key.shape}, tmp_value.shape={tmp_value.shape}")
    #                 # create the slotting_map for the tmp_key
                    
    #                 num_blocks_after_prune = (num_tmp_blocks - (prune_tokens // block_size))
    #                 if num_blocks_after_prune == 1:
    #                     block_idx = src_indices[0] 
    #                     tmp_slot_mapping = torch.arange(block_idx * block_size, (block_idx + 1) * block_size, 
    #                                 device=kv_cache_device, dtype=torch.long)
    #                 else: 
    #                     # num_blocks_after_prune > 1   
    #                     merge_block_ids = src_indices[:num_blocks_after_prune]
    #                     block_offsets = torch.arange(block_size, device=kv_cache_device, dtype=torch.long)
    #                     tmp_slot_mapping = (merge_block_ids[:, None] * block_size + block_offsets).flatten()
    #                 # end_3 = time.perf_counter()
    #                 pruned_keys.append(tmp_key)
    #                 pruned_values.append(tmp_value)
    #                 prunned_kvs_slot_mapping.append(tmp_slot_mapping)
    #                 # print(f"++++++{attn_metadata.decode_metadata.block_tables[id][s_idx_bt:e_idx_t]}, {merge_block_ids}")
    #                 # print(f"+++++++TorchCUDASDPABackendImpl---decode request: id={id}, seq_len={seq_len}, s_idx_bt={s_idx_bt}, e_idx_bt={e_idx_bt}, len(tmp_slot_mapping)={len(tmp_slot_mapping)}"\
    #                 #       f"create_kv={end_1 - start:.6f} prune_dur={end_2 - end_1:.6f} slot_dur={end_3 - end_2:.6f} seconds") 
        
    #     #Ensure All Active Streams Complete Before Returning Results
    #     for stream in active_streams.values():
    #         torch.cuda.current_stream().wait_stream(stream)

    #     # print(f"+++++++TorchCUDASDPABackendImpl---decode request cost={time.perf_counter() - start_1:.6f}")
    #     return pruned_keys, pruned_values, prunned_kvs_slot_mapping

    def _prune_decode_reqs_streamingLLM(self, kv_cache: torch.Tensor, attn_metadata: TorchCUDASDPAMetadata):
        # start_1 = time.perf_counter()
        kv_cache_device, kv_cache_dtype = kv_cache.device, kv_cache.dtype
        pin_memory = kv_cache_device.type == 'cpu'  # Use pinned memory for CPU cache
        
        pruned_keys, pruned_values, pruned_kvs_slot_mapping, is_prune_decode = None, None, None, False
        updated_block_tables, seqid_to_rmv_block_idx = None, {} 
        
        is_pruned_decode = False
        # upt_block_tables = [] 
        idx_to_rmv_block_idx = {}
        tmp_slot_mapping = []
        for id, seq_len in enumerate(attn_metadata.seq_lens):
            target_req_id, target_seq_id = self.block_l2norm_mgr.get_reqid_and_seqid(id)
            # Step-1: check if the request reach to the prunning point
            if seq_len <= self.paged_evict_config.cache_budget:
                # upt_block_tables.append(self.block_l2norm_mgr.get_req_block_tables(id))
                continue
            
            # seq_len exceed the cache budget, se we need to prune tokens
            if seq_len % self.cache_config.block_size != 0:
                if self.paged_evict_config.evict_method == "streamingLLM-1":
                    # For streamingLLM-1, we only prune the 2nd block
                    block_id = attn_metadata.reqid_mapping_blocktables[target_req_id][int(target_seq_id)][1]
                    slot_idx = seq_len % self.cache_config.block_size - 1
                    slot_id = block_id * self.cache_config.block_size + slot_idx
                    tmp_slot_mapping.append(slot_id)
                # upt_block_tables.append(self.block_l2norm_mgr.get_req_block_tables(id))
            else:
                seqid_to_rmv_block_idx[id] = (target_req_id, 1)
                idx_to_rmv_block_idx[id] = 1
                # upt_block_tables.append(self.block_l2norm_mgr.get_updated_block_tables(id, 1))
                is_pruned_decode = True

        num_update_tokens = len(tmp_slot_mapping)
        if num_update_tokens > 0:
        #    pruned_keys = torch.zeros((num_update_tokens, self.num_kv_heads, self.head_size), 
        #                                 dtype=kv_cache_dtype, device=kv_cache_device, pin_memory=pin_memory)
        #    pruned_values = torch.zeros((num_update_tokens, self.num_kv_heads, self.head_size), 
        #                                 dtype=kv_cache_dtype, device=kv_cache_device, pin_memory=pin_memory)
        #    pruned_kvs_slot_mapping = torch.tensor(tmp_slot_mapping,
        #                                             dtype=torch.long,
        #                                             device=kv_cache_device, pin_memory=pin_memory)
            pruned_kvs_slot_mapping = torch.from_numpy(np.array(tmp_slot_mapping, dtype=np.int64)).to(
                device=kv_cache_device, non_blocking=True)    
        
        if is_pruned_decode:
            # updated_block_tables = make_tensor_with_pad(
            #         upt_block_tables,
            #         pad=0,
            #         dtype=torch.int,
            #         device=kv_cache_device,
            #     )
            updated_block_tables = self._create_updated_block_tables(
                attn_metadata.decode_metadata.block_tables, idx_to_rmv_block_idx)
                 
        # print(f"+++++++TorchCUDASDPABackendImpl---decode request cost={time.perf_counter() - start_1:.6f}")
        # return pruned_keys, pruned_values, pruned_kvs_slot_mapping, is_pruned_decode, updated_block_tables, seqid_to_rmv_block_idx
        return num_update_tokens, pruned_kvs_slot_mapping, is_pruned_decode, updated_block_tables, seqid_to_rmv_block_idx

    def _prune_decode_reqs_inverseKeyL2(self, kv_cache: torch.Tensor, attn_metadata: TorchCUDASDPAMetadata):
        # start_1 = time.perf_counter()
        kv_cache_device, kv_cache_dtype = kv_cache.device, kv_cache.dtype
        pin_memory = kv_cache_device.type == 'cpu'  # Use pinned memory for CPU cache
            
        pruned_keys, pruned_values, pruned_kvs_slot_mapping = None, None, None
        
        is_pruned_decode = False
        tmp_slot_mapping = []
        for id, seq_len in enumerate(attn_metadata.seq_lens):
            # Step-1: check if the request reach to the prunning point
            if seq_len <= self.paged_evict_config.cache_budget:
                continue
            # randomly select a block id and a slot id and set the value of that one to 0
            target_req_id, target_seq_id = self.block_l2norm_mgr.get_reqid_and_seqid(id)
            block_ids = attn_metadata.reqid_mapping_blocktables[target_req_id][int(target_seq_id)]
            rnd_block_idx = random.randint(1, len(block_ids) - 2) # select block from the middle (except first and last block)
            rnd_slot_idx = random.randint(0, self.cache_config.block_size - 1)
            slot_id = block_ids[rnd_block_idx] * self.cache_config.block_size + rnd_slot_idx
            tmp_slot_mapping.append(slot_id)
        
        # create the prunned keys and values based on the tmp_slot_mapping
        num_update_tokens = len(tmp_slot_mapping)
        if num_update_tokens > 0:
            # t_start = time.perf_counter()
            # pruned_keys = torch.zeros((num_update_tokens, self.num_kv_heads, self.head_size), 
            #                             dtype=kv_cache_dtype, device=kv_cache_device, pin_memory=pin_memory)
            # pruned_values = torch.zeros((num_update_tokens, self.num_kv_heads, self.head_size), 
            #                             dtype=kv_cache_dtype, device=kv_cache_device, pin_memory=pin_memory)
            # pruned_keys = self.tmp_cache_singleton.get_cached_keys(num_update_tokens)
            # pruned_values = self.tmp_cache_singleton.get_cached_values(num_update_tokens)
            pruned_kvs_slot_mapping = torch.from_numpy(np.array(tmp_slot_mapping, dtype=np.int64)).to(
                device=kv_cache_device, non_blocking=True)
            # pruned_kvs_slot_mapping = torch.tensor(tmp_slot_mapping,
            #                                         dtype=torch.long,
            #                                         device=kv_cache_device, pin_memory=pin_memory)
            # t_end = time.perf_counter() - t_start
            # logger.info(f"Create prunned keys and values cost {t_end:.6f} seconds. num_update_tokens={num_update_tokens}, pruned_keys.shape={pruned_keys.shape}, pruned_values.shape={pruned_values.shape}, pruned_kvs_slot_mapping.shape={pruned_kvs_slot_mapping.shape}")
        
        # return pruned_keys, pruned_values, pruned_kvs_slot_mapping, is_pruned_decode
        return num_update_tokens, pruned_kvs_slot_mapping, is_pruned_decode 
    
    
    def _prune_decode_reqs_topk(self, kv_cache: torch.Tensor, attn_metadata: TorchCUDASDPAMetadata):
        # start_1 = time.perf_counter()
        kv_cache_device, kv_cache_dtype = kv_cache.device, kv_cache.dtype
        pin_memory = kv_cache_device.type == 'cpu'
        
        # pruned_keys, pruned_values, prunned_kvs_slot_mapping, updated = None, None, None, 
        updated_block_tables, seqid_to_rmv_block_idx = None, {} 
        
        is_pruned_decode = False
        idx_to_rmv_block_idx = {}
        # upt_block_tables = []
        for id, seq_len in enumerate(attn_metadata.seq_lens):
            # Step-1: check if the request reach to the prunning point
            if seq_len <= self.paged_evict_config.cache_budget:
                # upt_block_tables.append(self.block_l2norm_mgr.get_req_block_tables(id))
                continue
            
            # step-2: check if seq_len reach to the boundary of a block
            if seq_len % self.cache_config.block_size != 0:
                # upt_block_tables.append(self.block_l2norm_mgr.get_req_block_tables(id))
                continue
            
            is_pruned_decode = True
            target_req_id, target_seq_id = self.block_l2norm_mgr.get_reqid_and_seqid(id)
            if self.enable_random_evict == True:
                # randomly select a block id to be removed
                block_ids = attn_metadata.reqid_mapping_blocktables[target_req_id][int(target_seq_id)]
                if self.paged_evict_config.evict_method == "local":
                    blk_start = 1
                    blk_end = blk_start + self.paged_evict_config.topk_blocks
                else:
                    # The global case
                    blk_start = 1
                    blk_end = len(block_ids) - 1

                rmv_idx = random.randint(blk_start, blk_end - 1) # Exclude the last block
                seqid_to_rmv_block_idx[id] = (target_req_id,rmv_idx)
                idx_to_rmv_block_idx[id] = rmv_idx
                # upt_block_tables.append(self.block_l2norm_mgr.get_updated_block_tables(id, rmv_idx, is_rmv_l2norm=False))
            else: 
                # step-3: calculate the value-l2-norm of the last block
                # src_idx = attn_metadata.decode_metadata.block_tables[id][-1]
                src_idx = attn_metadata.reqid_mapping_blocktables[target_req_id][int(target_seq_id)][-1]
                tmp_kvs = kv_cache[:, src_idx, :]
                tmp_key = tmp_kvs[0].view(-1, self.num_kv_heads, self.head_size)
                tmp_value = tmp_kvs[1].view(-1, self.num_kv_heads, self.head_size)
                l2_norm = self.kv_cache_pruner.get_block_score(tmp_key, tmp_value, self.paged_evict_config.evict_method)
                # update the block l2-norm value
                l2_norms = self.block_l2norm_mgr.update_last_block_l2norm(id, l2_norm)
                # get the removed block idx
                if self.paged_evict_config.evict_method == "local":
                    blk_start = 1
                    blk_end = blk_start + self.paged_evict_config.topk_blocks
                else:
                    ### The global case
                    blk_start = 1
                    blk_end = len(l2_norms) - 1 # not include the last block
                    # get the idx that has the minimal l2-norm
                rmv_idx = blk_start
                min_l2_norm = l2_norms[rmv_idx]
                for idx in range(blk_start, blk_end):
                    if l2_norms[idx] < l2_norms[rmv_idx]:
                        rmv_idx = idx
                        min_l2_norm = l2_norms[idx]
                
                seqid_to_rmv_block_idx[id] = (target_req_id, rmv_idx)
                idx_to_rmv_block_idx[id] = rmv_idx 

                # upt_block_tables.append(self.block_l2norm_mgr.get_updated_block_tables(id, rmv_idx, is_rmv_l2norm=True))
                # print(f"layer_idx = {self.layer_idx} raw block tables type: {type(attn_metadata.decode_metadata.block_tables)}, shape={attn_metadata.decode_metadata.block_tables.shape} raw block tables: {attn_metadata.decode_metadata.block_tables[id]}, updated block tables: {upt_block_tables[id]}")
 
        ### Create a new block tables based on the upt_block_tables
        if is_pruned_decode:
            # updated_block_tables = make_tensor_with_pad(
            #         upt_block_tables,
            #         pad=0,
            #         dtype=torch.int,
            #         device=kv_cache_device,
            #     )
            updated_block_tables = self._create_updated_block_tables(
                attn_metadata.decode_metadata.block_tables, idx_to_rmv_block_idx) 
         
        return updated_block_tables, seqid_to_rmv_block_idx, is_pruned_decode
    
    def _create_updated_block_tables(self, original_tables, idx_to_rmv_block_idx):
        # This function is used to create the updated block tables for the streamingLLM case
        batch_size, max_blocks = original_tables.shape
        device = original_tables.device
        
        # clone the original tables, we cannot modify the original tables
        # because it will be used in the next decode layer
        updated_tables = original_tables.clone()
        max_new_length = max_blocks - 1  # Get the sequence IDs from the first column
        for idx in range(batch_size):
            origin_blocks = updated_tables[idx]
            if idx in idx_to_rmv_block_idx:
                rmv_block_idx = idx_to_rmv_block_idx[idx]
                # Remove the block at rmv_block_idx and shift the rest left
                new_blocks = torch.cat([origin_blocks[:rmv_block_idx], origin_blocks[rmv_block_idx+1:]])
            else:
                # No block to remove, keep the original blocks
                new_blocks = origin_blocks[:max_new_length]
            
            # write the new blocks to the updated tables
            updated_tables[idx, :len(new_blocks)] = new_blocks
            updated_tables = updated_tables[:,:max_new_length]  # Ensure the shape is consistent
        
        # print(f"+++++create_updated_block_tables: original_tables.shape={original_tables.shape}, updated_tables.shape={updated_tables.shape}, max_new_length={max_new_length}")    
        return updated_tables


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    dtype: torch.dtype,
    seq_lens: List[int],
) -> List[torch.Tensor]:
    attn_biases: List[torch.Tensor] = []
    for seq_len in seq_lens:
        bias = torch.arange(seq_len, dtype=dtype)
        # NOTE(zhuohan): HF uses
        #     `bias = bias[None, :].repeat(seq_len, 1)`
        # here. We find that both biases give the same results, but
        # the bias below more accurately follows the original ALiBi
        # paper.
        bias = bias[None, :] - bias[:, None]

        num_heads = alibi_slopes.shape[0]
        bias = bias[None, :].repeat((num_heads, 1, 1))
        bias.mul_(alibi_slopes[:, None, None]).unsqueeze_(0)
        inf_mask = torch.empty(
            (1, seq_len, seq_len),
            dtype=bias.dtype).fill_(-torch.inf).triu_(diagonal=1)
        attn_biases.append((bias + inf_mask).to(dtype))

    return attn_biases


def _make_sliding_window_bias(
    seq_lens: List[int],
    window_size: Optional[int],
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    attn_biases: List[torch.Tensor] = []
    for seq_len in seq_lens:
        tensor = torch.full(
            (1, seq_len, seq_len),
            dtype=dtype,
            fill_value=1,
        )
        shift = 0
        mask = torch.tril(tensor, diagonal=shift).to(dtype)  # type: ignore
        if window_size is not None:
            mask = torch.triu(mask, diagonal=shift - window_size + 1)
        mask = torch.log(mask)
        attn_biases.append(mask.to(dtype))

    return attn_biases