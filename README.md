## 1. Install vLLM
#### Step_1: Install the request packages
```
pip install -r requirements-build.txt
```
If you want to use existing pytorch in your environment:
```
python use_existing_torch.py
pip install -r requirements-build.txt
```

#### Step_2: Install vLLM on Polaris
```
python -m pip install --ignore-installed vllm --no-build-isolation -vvv . --no-deps
```

# 2. Running Inferences
#### Step_1: Configure environment variables

We only support TORCH_SDPA Attention right now, so you need to configure the following environment before running inference (On Polaris):
```
IF_NAME=hsn0
export HOST_IP=`ip -4 addr show $IF_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}'`
export VLLM_HOST_IP=$HOST_IP
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_RPC_BASE_PATH=/tmp/vllm_ipc/
```


#### Step_2: Running inferences manually with the commandline (Optional)
1. Running inference without PagedCompression
    ```
    python $HOME/vllm-Paged-Compression/benchmarks/benchmark_throughput.py --backend vllm --input-len 1024 --output-len 100 --num-prompts 1 --model $MODEL_PATH --enforce-eager --distributed-executor-backend mp --tensor-parallel-size 1
    ```

2. Running inference with PagedCompression (Percentage)

    2.1 When the eviction rate is 25%:
    ```
    python $HOME/vllm-Paged-Compression/benchmarks/benchmark_throughput.py --backend vllm --input-len 1024 --output-len 100 --num-prompts 1 --model $MODEL_PATH --enforce-eager --distributed-executor-backend mp --tensor-parallel-size 1 --enable-paged-eviction --cache-prune-type percentage --prompt-evict-method streamingLLM --decode-evict-method value_l2 --block-size 16 --evict-freq 2
    ``` 
    2.2. When the eviction rate is 50%:
    ```
    python $HOME/vllm-Paged-Compression/benchmarks/benchmark_throughput.py --backend vllm --input-len 1024 --output-len 100 --num-prompts 1 --model $MODEL_PATH --enforce-eager --distributed-executor-backend mp --tensor-parallel-size 1 --enable-paged-eviction --cache-prune-type percentage --prompt-evict-method streamingLLM --decode-evict-method value_l2 --block-size 16 --evict-freq 4
    ```

### Step_3: Running inferences manually with the scripts
All the scripts are stored in the `scripts` folder.

1. Benchmark the inference performance by varying the number of requets when serving requests with and without pagedcompression (the input/output length is fixed):

    A. Modify the `scripts/run_benchmark_full.sh` to configure the conresponding parameters: model path, and log path
        
    B. Run the script
    ```
    cd $HOME/vllm-Paged-Compression/scripts/
    ./run_benchmark_full.sh
    ```

2. Benchmark the inference performance by varying the input/output length when running with and without pagedcompression:

    A. Modify the `scripts/diff_seqlens_full.sh` to configure the conresponding parameters: model path, and log path
        
    B. Run the script 
    ```
    cd $HOME/vllm-Paged-Compression/scripts/
    ./diff_seqlens_full.sh
    ```

