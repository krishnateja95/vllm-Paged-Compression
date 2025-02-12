#!/bin/bash
USE_EVICT_FREQ=1
# import env vars
source $HOME/acl25/vllm_env_vars

# COMMON PARAMETERS
INPUT_LEN=1024
OUTPUT_LEN=8192
NUM_REQS=(10 20 30 40 50 60 70 80 90 100)
#NUM_REQS=(80)

# Model PATH
MODEL_PATH="/eagle/projects/RECUP/jye/huggingface-hub/"
### Models to test
MODELS=("Llama-3.1-8B-Instruct" "Mistral-7B-Instruct-v0.2" "Qwen2.5-7B-Instruct-1M")
### Cache types to test
CACHE_TYPES=("percentage" "full-cache")


BLOCK_SIZES=(8 16 32)

TP=1
BENCHMARK_SCRIPT_DIR="$HOME/acl25/vllm-Paged-Compression-v3/benchmarks/"
LOG_DIR="$HOME/acl25/performance_logs/various_reqs_tp${TP}/"

#################################################################################
# Helper functions
create_log_dir() {
    local log_dir=$1
    [ ! -d "$log_dir" ] && mkdir -p "$log_dir"
}

run_percentage_benchmark_with_evictsize() {
    local MODEL=$1 CACHE_TYPE=$2 P_EVICT_METHOD=$3 D_EVICT_METHOD=$4 NUM_REQ=$5 BLOCK_SIZE=$6 EVICT_SIZE=$7 TMP_LOG_DIR=$8 LOG_FILE_NAME=$9
    local MODEL_NAME="${MODEL_PATH}${MODEL}"
    local max_model_len=""

    echo -e "\tProcessing $NUM_REQ requests with $MODEL model using prompt_evict_method($P_EVICT_METHOD) and decode_evict_method($D_EVICT_METHOD)..."
    echo -e "\t\tnum_req: $NUM_REQ, block_size: $BLOCK_SIZE, evict_size: $EVICT_SIZE"
    echo -e "\t\tLog file: ${TMP_LOG_DIR}${LOG_FILE_NAME}"
    echo -e "\tStart run the percentage benchmark...."

    [ "$MODEL" = "Qwen2.5-7B-Instruct-1M" ] && max_model_len="--max-model-len 131072"

    python3 $BENCHMARK_SCRIPT_DIR/benchmark_throughput.py \
            --backend vllm \
            --model $MODEL_NAME \
            --enforce-eager \
            --distributed-executor-backend mp \
            --tensor-parallel-size $TP \
            --input-len $INPUT_LEN \
            --output_len $OUTPUT_LEN \
            --num-prompts $NUM_REQ \
            --enable-paged-eviction \
            --cache-prune-type $CACHE_TYPE \
            --prompt-evict-method $P_EVICT_METHOD \
            --decode-evict-method $D_EVICT_METHOD \
            --block-size $BLOCK_SIZE \
            --evict_size $EVICT_SIZE \
            --initial-blocks 1 \
            --gpu-memory-utilization 0.9 \
            $max_model_len \
            > ${TMP_LOG_DIR}${LOG_FILE_NAME} 2>&1
    echo "Write to log file: ${TMP_LOG_DIR}${LOG_FILE_NAME}"

    echo -e "\tDone"
}

run_percentage_benchmark_with_evictfreq() {
    local MODEL=$1 CACHE_TYPE=$2 P_EVICT_METHOD=$3 D_EVICT_METHOD=$4 NUM_REQ=$5 BLOCK_SIZE=$6 EVICT_FREQ=$7 TMP_LOG_DIR=$8 LOG_FILE_NAME=$9
    local MODEL_NAME="${MODEL_PATH}${MODEL}"
    local max_model_len=""

    echo -e "\tProcessing $NUM_REQ requests with $MODEL model using prompt_evict_method($P_EVICT_METHOD) and decode_evict_method($D_EVICT_METHOD)..."
    echo -e "\t\tnum_req: $NUM_REQ, block_size: $BLOCK_SIZE, evict_freq: $EVICT_FREQ"
    echo -e "\t\tLog file: ${TMP_LOG_DIR}${LOG_FILE_NAME}"
    echo -e "\tStart run the percentage benchmark...."

    [ "$MODEL" = "Qwen2.5-7B-Instruct-1M" ] && max_model_len="--max-model-len 131072"

    # Currently only support num_blocks_merge=2
    python3 $BENCHMARK_SCRIPT_DIR/benchmark_throughput.py \
            --backend vllm \
            --model $MODEL_NAME \
            --enforce-eager \
            --distributed-executor-backend mp \
            --tensor-parallel-size $TP \
            --input-len $INPUT_LEN \
            --output_len $OUTPUT_LEN \
            --num-prompts $NUM_REQ \
            --enable-paged-eviction \
            --cache-prune-type $CACHE_TYPE \
            --prompt-evict-method $P_EVICT_METHOD \
            --decode-evict-method $D_EVICT_METHOD \
            --block-size $BLOCK_SIZE \
            --evict_freq $EVICT_FREQ \
            --initial-blocks 1 \
            --num-blocks-merge 2 \
            --gpu-memory-utilization 0.9 \
            $max_model_len \
            > ${TMP_LOG_DIR}${LOG_FILE_NAME} 2>&1
    echo "Write to log file: ${TMP_LOG_DIR}${LOG_FILE_NAME}"
    echo -e "\tDone"
}

run_fullcache_bechnamrk() {
    local MODEL=$1 NUM_REQ=$2 BLOCK_SIZE=$3 TMP_LOG_DIR=$4 LOG_FILE_NAME=$5
    local MODEL_NAME="${MODEL_PATH}${MODEL}"
    local max_model_len=""

    echo -e "\tProcessing $NUM_REQ requests with $MODEL model..."
    echo -e "\t\tLog file: ${TMP_LOG_DIR}${LOG_FILE_NAME}"
    echo -e "\tStart run the benchmark...."

    [ "$MODEL" = "Qwen2.5-7B-Instruct-1M" ] && max_model_len="--max-model-len 131072"

    python3 $BENCHMARK_SCRIPT_DIR/benchmark_throughput.py \
            --backend vllm \
            --model $MODEL_NAME \
            --enforce-eager \
            --distributed-executor-backend mp \
            --tensor-parallel-size $TP \
            --input-len $INPUT_LEN \
            --output_len $OUTPUT_LEN \
            --num-prompts $NUM_REQ \
            --gpu-memory-utilization 0.9 \
            ${max_model_len} \
            > ${TMP_LOG_DIR}${LOG_FILE_NAME} 2>&1
    echo "Write to log file: ${TMP_LOG_DIR}${LOG_FILE_NAME}"

    echo -e "\tDone"
}

#################################################################################

echo "Start the benchmark..."
echo "===================================================================="

for CACHE_TYPE in "${CACHE_TYPES[@]}"; do
    if [ "$CACHE_TYPE" = "percentage" ]; then
        echo "Start the percentage benchmark test ..."
        
        PROMPT_AND_DECODE_EVICT_METHODS=(
            "streamingLLM streamingLLM"
            "streamingLLM value_l2"
            "value_l2 value_l2"
            "value_l2 streamingLLM"
        )

        if [ $USE_EVICT_FREQ -eq 1 ]; then
            EVICT_FREQS=(2 4)
        else
            BLOCK_SIZE_AND_EVICT_SIZE_PAIR=(
                "16 8"
                "32 16"
                "64 32"
            )
        fi

        for P_AND_D_EVICT_METHOD in "${PROMPT_AND_DECODE_EVICT_METHODS[@]}"; do
            read -r P_EVICT_METHOD D_EVICT_METHOD <<< "$P_AND_D_EVICT_METHOD" 
            TMP_LOG_DIR="${LOG_DIR}${CACHE_TYPE}/${P_EVICT_METHOD}_${D_EVICT_METHOD}/"
            # check if the TMP_LOG_DIR1 exists; if not, create it
            create_log_dir "$TMP_LOG_DIR"
            echo "Running benchmarks with $CACHE_TYPE cache type using prompt_evict_method=$P_EVICT_METHOD, decode_evict_method=$D_EVICT_METHOD"

            for NUM_REQ in "${NUM_REQS[@]}"; do
                echo -e "\tStart the test for request $NUM_REQ...."
                for MODEL in "${MODELS[@]}"; do
                    echo -e "\Start the test with $MODEL model..." 
                    if [ $USE_EVICT_FREQ -eq 1 ]; then
                        for BLOCK_SIZE in "${BLOCK_SIZES[@]}"; do
                            for EVICT_FREQ in "${EVICT_FREQS[@]}"; do
                                LOG_FILE_NAME="result_m${MODEL}_b${BLOCK_SIZE}_ef${EVICT_FREQ}_r${NUM_REQ}_p${INPUT_LEN}_g${OUTPUT_LEN}_tp${TP}.log"
                                run_percentage_benchmark_with_evictfreq "$MODEL" "$CACHE_TYPE" "$P_EVICT_METHOD" "$D_EVICT_METHOD" "$NUM_REQ" "$BLOCK_SIZE" "$EVICT_FREQ" "$TMP_LOG_DIR" "$LOG_FILE_NAME"
                                sleep 1
                            done
                        done
                    else
                        for BLOCK_SIZE_AND_EVICT_SIZE in "${BLOCK_SIZE_AND_EVICT_SIZE_PAIR[@]}"; do
                            read -r BLOCK_SIZE EVICT_SIZE <<< "$BLOCK_SIZE_AND_EVICT_SIZE"
                            LOG_FILE_NAME="result_m${MODEL}_b${BLOCK_SIZE}_e${EVICT_SIZE}_r${NUM_REQ}_p${INPUT_LEN}_g${OUTPUT_LEN}_tp${TP}.log"
                            run_percentage_benchmark_with_evictsize "$MODEL" "$CACHE_TYPE" "$P_EVICT_METHOD" "$D_EVICT_METHOD" "$NUM_REQ" "$BLOCK_SIZE" "$EVICT_SIZE" "$TMP_LOG_DIR" "$LOG_FILE_NAME"
                            sleep 1
                        done
                    fi
                    echo -e "\tFinished the test with $MODEL model..." 
                done
                echo -e "\tFinished the test for request $NUM_REQ with $P_EVICT_METHOD and $D_EVICT_METHOD evict method with $CACHE_TYPE cache type..." 
            done
        done 
    elif [ "$CACHE_TYPE" = "full-cache" ]; then
        TMP_LOG_DIR="${LOG_DIR}${CACHE_TYPE}/"
        # check if the TMP_LOG_DIR1 exists; if not, create it
        create_log_dir "$TMP_LOG_DIR"
        echo "Running benchmarks with $CACHE_TYPE cache type"

        for NUM_REQ in "${NUM_REQS[@]}"; do
            echo -e "\tStart the test of request $NUM_REQ...."  
            for MODEL in "${MODELS[@]}"; do
                echo -e "\Start the test with $MODEL model..."  
                for BLOCK_SIZE in "${BLOCK_SIZES[@]}"; do
                    LOG_FILE_NAME="result_m${MODEL}_b${BLOCK_SIZE}_r${NUM_REQ}_p${INPUT_LEN}_g${OUTPUT_LEN}_tp${TP}.log"
                    run_fullcache_bechnamrk "$MODEL" "$NUM_REQ" "$BLOCK_SIZE" "$TMP_LOG_DIR" "$LOG_FILE_NAME"
                    sleep 1
                done
                echo -e "\tFinished the test with $MODEL model..." 
            done
            echo -e "\tFinished the test of request $NUM_REQ with $CACHE_TYPE cache type..."  
        done
        echo "Finished the test with $CACHE_TYPE cache type..."
    else
        echo "ERROR: Unknown cache type $CACHE_TYPE"
    fi
    echo "===================================================================="
done

echo "Finished the benchmark..."
