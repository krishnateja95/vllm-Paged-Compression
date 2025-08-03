#!/bin/bash
# import env vars
source $HOME/kv_evict/vllm-Paged-Compression/scripts/vllm_env_vars

# Model PATH
MODEL_PATH="/vast/users/jye/huggingface-hub/"
### Models to test
# MODELS=("meta-llama/Llama-3.2-3B-Instruct" "meta-llama/Llama-3.2-1B-Instruct" "meta-llama/Llama-3.1-8B-Instruct")
MODELS=("meta-llama/Llama-3.1-8B-Instruct")

# EVICT_METHODS=("default" "inverse_key_l2" "streamingLLM-1" "local" "global" "streamingLLM")
EVICT_METHODS=("streamingLLM-1" "local" "global" "inverse_key_l2")

CACHE_BUDGET=1024
### KV cache page sizes to test
PAGE_SIZES=(8 16 32)

# COMMON PARAMETERS
INPUT_OUTPUT_PAIRS=(
    # "1024 1024"
    # "2048 2048"
    # "4096 4096"
    # "8192 8192"
    # "16384 16384"
    "1024 8192"
)

NUM_REQS=(64)
TOPK_BLOCKS=3
DISABLE_EVICT_PREFILL=True # True: disable, False: enable
EXEC_PATH="$HOME/kv_evict/vllm-Paged-Compression/benchmarks/"
BASE_LOG_PATH="$HOME/kv_evict/perf_logs/diff_pages_v0_1/"

#################################################################################
# Helper functions
create_log_dir() {
    local log_dir=$1
    [ ! -d "$log_dir" ] && mkdir -p "$log_dir"
}

run_kv_evict_benchmarks_with_pagesize() {
    local model=$1 evict_method=$2 cache_budget=$3 page_size=$4 num_req=$5 input_len=$6 output_len=$7 evict_log_path=$8 log_file_name=$9
    local model_name=`echo ${model} | cut -d '/' -f 2`
    local evict_topk_blocks=$TOPK_BLOCKS

    echo -e "\tProcessing $num_req requests with $model model using evict_method($evict_method)..."
    echo -e "\tnum_req: $num_req, input_len: $input_len, output_len: $output_len, page_size: $page_size"

    if [ "$evict_method" = "local" ]; then
        echo -e "\tUsing local eviction method with topk-blocks=$evict_topk_blocks"
    elif [ "$evict_method" = "global" ]; then
        evict_topk_blocks=-1
        echo -e "\tUsing global eviction method with topk-blocks=$evict_topk_blocks"
    fi

    echo -e "\tLog file: ${evict_log_path}${log_file_name}"
    echo -e "\tStart run the benchmark with ${evict_method} evict method...."

    exec_cmd="python3 $EXEC_PATH/benchmark_throughput.py \
            --backend vllm \
            --model $model \
            --download-dir ${MODEL_PATH} \
            --enforce-eager \
            --distributed-executor-backend mp \
            --tensor-parallel-size 1 \
            --input-len $input_len \
            --output_len $output_len \
            --num-prompts $num_req \
            --enable-paged-eviction \
            --cache-prune-type budget \
            --evict-method $evict_method \
            --cache-budget $cache_budget \
            --gpu-memory-utilization 0.9 \
            --block-size $page_size \
            --topk-blocks ${evict_topk_blocks} "
    
    if [ "$DISABLE_EVICT_PREFILL" = "True" ]; then
        exec_cmd+=" --disable-evict-prefill "
    fi
    # echo -e "\tExecuting command: $exec_cmd"

    eval "$exec_cmd" > "${evict_log_path}${log_file_name}" 2>&1

    sleep 5

    echo -e "Benchmark with ${evict_method} evict method Done"
}

run_fullcache_benchmark() {
    local model=$1 page_size=$2 num_req=$3 input_len=$4 output_len=$5 full_cache_log_path=$6 log_file_name=$7
    local model_name=`echo ${model} | cut -d '/' -f 2`

    echo -e "\tProcessing $num_req requests (page_size=$page_size, inputlen=$input_len, outputlen=$output_len) with $model model..."
    echo -e "\tLog file: ${full_cache_log_path}${log_file_name}"
    echo -e "\tStart run the baseline benchmark (i.e., the full cache)...."

    python3 $EXEC_PATH/benchmark_throughput.py \
            --backend vllm \
            --model $model \
            --download-dir ${MODEL_PATH} \
            --enforce-eager \
            --distributed-executor-backend mp \
            --tensor-parallel-size 1 \
            --input-len $input_len \
            --output_len $output_len \
            --num-prompts $num_req \
            --gpu-memory-utilization 0.9 \
            --block-size $page_size \
            > ${full_cache_log_path}${log_file_name} 2>&1

    sleep 5

    echo -e "Baseline benchmark with inputlen=${input_len} outputlen=${output_len} Done"
}

#################################################################################

echo "Start test vLLM's throughput performance with full cache and different eviction methods"
echo "============================================================================================"

for evict_method in "${EVICT_METHODS[@]}"; do
    if [ "$evict_method" = "default" ]; then
        #### Test the throughput performance with full cache
        echo "Start the benchmark with full cache (i.e., KV cache eviction disabled)..."
        for model in "${MODELS[@]}"; do
            model_name=`echo ${model} | cut -d '/' -f 2`
            for num_req in "${NUM_REQS[@]}"; do
                for INPUT_OUTPUT_LEN in "${INPUT_OUTPUT_PAIRS[@]}"; do
                    IFS=" " read -r input_len output_len <<< "$INPUT_OUTPUT_LEN"
                    full_cache_log_path="${BASE_LOG_PATH}/full_cache/"
                    create_log_dir "$full_cache_log_path"
                    for page_size in "${PAGE_SIZES[@]}"; do
                        log_file_name="${model_name}_r${num_req}_p${input_len}_g${output_len}_ps${page_size}.log"
                        run_fullcache_benchmark "$model" "$page_size" "$num_req" "$input_len" "$output_len" "$full_cache_log_path" "$log_file_name"

                    done
                done
            done
            echo "==================================================================="
        done
    else
        #### Test the throughput performance with different eviction methods
        if [ "$DISABLE_EVICT_PREFILL" = "True" ]; then
            sub_dir="disable_prefill_evict"
        else
            sub_dir="enable_prefill_evict"
        fi
        echo "Start the benchmark with $evict_method eviction method..."
        for model in "${MODELS[@]}"; do
            model_name=`echo ${model} | cut -d '/' -f 2`
            for num_req in "${NUM_REQS[@]}"; do
                for INPUT_OUTPUT_LEN in "${INPUT_OUTPUT_PAIRS[@]}"; do
                    IFS=" " read -r input_len output_len <<< "$INPUT_OUTPUT_LEN"
                    evict_log_path="${BASE_LOG_PATH}/${sub_dir}/${evict_method}/"
                    create_log_dir "$evict_log_path"
                    cache_budget=${CACHE_BUDGET}
                    for page_size in "${PAGE_SIZES[@]}"; do
                        log_file_name="${model_name}_bd${cache_budget}_r${num_req}_p${input_len}_g${output_len}_ps${page_size}.log"
                        if [ "$evict_method" = "local" ]; then
                            log_file_name="${model_name}_bd${cache_budget}_r${num_req}_p${input_len}_g${output_len}_k${TOPK_BLOCKS}_ps${page_size}.log"
                        else
                            log_file_name="${model_name}_bd${cache_budget}_r${num_req}_p${input_len}_g${output_len}_ps${page_size}.log"
                        fi
                        run_kv_evict_benchmarks_with_pagesize "$model" "$evict_method" "$cache_budget" "$page_size" "$num_req" "$input_len" "$output_len" "$evict_log_path" "$log_file_name"
                        echo "********************************************************************"
                    done 
                done
            done
            echo "==================================================================="
        done
    fi
done

echo "Finished the benchmark..."
