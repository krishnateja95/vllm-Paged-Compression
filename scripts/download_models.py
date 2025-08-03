from transformers import AutoModel,AutoModelForCausalLM
import huggingface_hub.constants
from huggingface_hub import snapshot_download
import filelock
from tqdm.auto import tqdm
from typing import Optional
import os
import hashlib
import tempfile

temp_dir = tempfile.gettempdir()

class DisabledTqdm(tqdm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)
        
def get_lock(model_name_or_path: str, cache_dir: Optional[str] = None):
    lock_dir = cache_dir or temp_dir
    os.makedirs(os.path.dirname(lock_dir), exist_ok=True)
    model_name = model_name_or_path.replace("/", "-")
    hash_name = hashlib.sha256(model_name.encode()).hexdigest()
    # add hash to avoid conflict with old users' lock files
    lock_file_name = hash_name + model_name + ".lock"
    # mode 0o666 is required for the filelock to be shared across users
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name),
                             mode=0o666)
    return lock
        

task="download_model" # download_model or download_dataset
if task == "download_model":
    # model_name_or_path = f"meta-llama/Llama-3.3-70B-Instruct"
    # model_name_or_path = f"meta-llama/Llama-3.1-8B"
    # model_name_or_path = f"meta-llama/Llama-3.2-3B-Instruct"
    #model_name_or_path = f"meta-llama/Llama-3.2-1B-Instruct"
    model_name_or_path = f"meta-llama/Llama-3.1-8B-Instruct"
    #cache_dir = f"/lus/eagle/projects/RECUP/jye/huggingface-hub/"
    cache_dir = f"/vast/users/jye/huggingface-hub/"
    # cache_dir = f"/lus/grand/projects/VeloC/jye/viper2/huggingface-hub/"
    # allow_patterns = ["*.safetensors", "*.bin"]
    allow_patterns = ["*.safetensors", "*.bin", "*.json", "*.txt", '*.yml']
    ignore_patterns = ["original/**/*"]

    with get_lock(model_name_or_path, cache_dir):
        hf_folder = snapshot_download(
            model_name_or_path,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            cache_dir=cache_dir,
            tqdm_class=DisabledTqdm,
            revision=None,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
        )

elif task == "download_dataset":
    from datasets import load_dataset

    use_longbench_v2 = False
    
    cache_dir = "/lus/eagle/projects/RECUP/jye/huggingface-hub/"
    os.makedirs(cache_dir, exist_ok=True)
    if not use_longbench_v2:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
        for dataset in datasets:
            data = load_dataset('THUDM/LongBench', dataset, cache_dir=cache_dir, split='test')
    else:
        data = load_dataset('THUDM/LongBench-v2', cache_dir=cache_dir, split='train') 
else:
    raise ValueError(f"Unknown task: {task}")