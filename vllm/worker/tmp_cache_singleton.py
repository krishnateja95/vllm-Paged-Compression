import threading
from typing import Dict, Any, Optional
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
import torch


class TmpCacheSingleton:
    """
    Thread-safe singleton class for storing temporary key-value cache
    that can be shared across multiple classes.
    """
    _instance: Optional['TmpCacheSingleton'] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs) -> 'TmpCacheSingleton':
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, max_num_seqs, head_size, num_kv_heads, dtype, device):
        # Prevent re-initialization
        if self._initialized:
            return
        
        ## create the cached keys and values
        self._cached_keys = torch.zeros((max_num_seqs * 2, num_kv_heads, head_size), dtype=dtype, device=device)
        self._cached_values = torch.zeros((max_num_seqs * 2, num_kv_heads, head_size), dtype=dtype, device=device)
        self._cache_lock = threading.RLock()
        self._initialized = True
        print(f"TmpCacheSingleton initialized with max_num_seqs={max_num_seqs * 2}, "
              f"_cached_keys.shape={self._cached_keys.shape}, _cached_values.shape={self._cached_values.shape}, "
              f"dtype={dtype}, device={device} key_cache_size={self._cached_keys.numel() * self._cached_keys.element_size() / (1024 ** 2):.2f} MB, "
              f"value_cache_size={self._cached_values.numel() * self._cached_values.element_size() / (1024 ** 2):.2f} MB")

    @classmethod
    def get_instance(cls) -> Optional['TmpCacheSingleton']:
        """
        Get the singleton instance if it exists, otherwise return None.
        Use this method to retrieve an already initialized instance.
        """
        return cls._instance
    
    @classmethod
    def create_instance(cls, cache_config, model_config, parallel_config, device_config, scheduler_config) -> 'TmpCacheSingleton':
        """
        Create or get the singleton instance with the provided configurations.
        If instance already exists, configurations will be ignored.
        """
        head_size = model_config.get_head_size()
        num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        device = device_config.device_type if device_config else "cpu"
        max_num_seqs = scheduler_config.max_num_seqs if scheduler_config else None
        return cls(max_num_seqs, head_size, num_kv_heads, dtype, device)
    
    @classmethod
    def delete_instance(cls) -> None:
        """
        Delete the singleton instance and free all resources.
        Only the one create the instance can delete it.
        """
        with cls._lock:
            if cls._instance is not None:
                # Clear the cached tensors first
                if hasattr(cls._instance, '_cached_keys') and cls._instance._cached_keys is not None:
                    del cls._instance._cached_keys
                if hasattr(cls._instance, '_cached_values') and cls._instance._cached_values is not None:
                    del cls._instance._cached_values
                
                # Reset the singleton instance
                cls._instance = None
     
    def get_cached_keys(self, num_keys) -> Any:
        """Get cached keys."""
        return self._cached_keys[:num_keys]

    def get_cached_values(self, num_values) -> Any:
        """Get cached values."""
        return self._cached_values[:num_values]
