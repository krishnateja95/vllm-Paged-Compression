#!/bin/bash
# COMMON PARAMETERS
INPUT_LEN=1024
OUTPUT_LEN=8192
NUM_REQ=50

BLOCK_SIZE=16

CACHE_TYPES=("full-cache" "percentage")

### Models to test
MODELS=("Llama-3.1-8B-Instruct" "Mistral-7B-Instruct-v0.2")

PROMPT_AND_DECODE_EVICT_METHODS=(
    "streamingLLM streamingLLM"
    "value_l2 value_l2"
)

TP=1
BASE_LOG_DIR="$HOME/acl25/performance_logs/various_reqs_tp${TP}/"
OUTPUT_DIR="$HOME/csv_results/mem_util_tp${TP}/"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

#####################################################################################
# Helper functions
process_percentage_evictfreq_logs() {
    CACHE_TYPE=$1
    local EVICT_FREQS=(2)

    for P_AND_D_EVICT_METHOD in "${PROMPT_AND_DECODE_EVICT_METHODS[@]}"; do
        read -r P_EVICT_METHOD D_EVICT_METHOD <<< "$P_AND_D_EVICT_METHOD"
        EVICT_METHOD_DIR="${P_EVICT_METHOD}_${D_EVICT_METHOD}"
        EVICT_METHOD="${P_EVICT_METHOD}+${D_EVICT_METHOD}" 
        for MODEL in "${MODELS[@]}"; do
            for EVICT_FREQ in "${EVICT_FREQS[@]}"; do
                CSV_FILE="${OUTPUT_DIR}${EVICT_METHOD}_m${MODEL}_p${INPUT_LEN}_g${OUTPUT_LEN}_r${NUM_REQ}_b${BLOCK_SIZE}_ef${EVICT_FREQ}.csv"
                if [ ! -f $CSV_FILE ]; then 
                    printf "time,model_name,total_gpu_blocks,running_reqs,waiting_reqs,gpu_kv_cache_usage\n" > $CSV_FILE 
                fi
                LOG_FILE="${BASE_LOG_DIR}${CACHE_TYPE}/${EVICT_METHOD_DIR}/result_m${MODEL}_b${BLOCK_SIZE}_ef${EVICT_FREQ}_r${NUM_REQ}_p${INPUT_LEN}_g${OUTPUT_LEN}_tp${TP}.log"
                echo "Start process log file $LOG_FILE"
                if [ -f $LOG_FILE ]; then
                    # Check if the log file contains "Throughput" (indicating a successful run)
                    if grep -q "Throughput" $LOG_FILE; then
                        COUNTER=0
                        TOTAL_GPU_BLOCKS=$(grep "# GPU blocks:" "$LOG_FILE"|awk '{gsub(/,/, "", $(NF-4)); print $(NF-4)}')
                        # record the initial GPU KV cache usage
                        printf "0,${MODEL},${TOTAL_GPU_BLOCKS},0,0,0\n" >> $CSV_FILE
                        grep "GPU KV cache usage" "$LOG_FILE"|awk -v model="$MODEL" -v total_gpu_blocks="$TOTAL_GPU_BLOCKS" -v counter=0 '
                        {
                            gsub(/%,/, "", $(NF-5)); # Remove "%," from GPU KV cache
                            counter += 5; # Increment counter by 5 since the statistics interval are 5 seconds
                            print counter "," model "," total_gpu_blocks "," $(NF-17) "," $(NF-11) "," $(NF-5);
                        }' >> $CSV_FILE
                    else
                        echo "ERROR: Log file $LOG_FILE is not complete"
                    fi
                else
                    echo "ERROR: Log file $LOG_FILE does not exist"
                fi
            done
        done
    done 
}

process_fullcache_logs() {
    CACHE_TYPE=$1
    for MODEL in "${MODELS[@]}"; do
        CSV_FILE="${OUTPUT_DIR}fullcache_m${MODEL}_p${INPUT_LEN}_g${OUTPUT_LEN}_r${NUM_REQ}_b${BLOCK_SIZE}.csv"
        if [ ! -f $CSV_FILE ]; then 
            printf "time,model_name,total_gpu_blocks,running_reqs,waiting_reqs,gpu_kv_cache_usage\n" > $CSV_FILE 
        fi
        LOG_FILE="${BASE_LOG_DIR}${CACHE_TYPE}/result_m${MODEL}_b${BLOCK_SIZE}_r${NUM_REQ}_p${INPUT_LEN}_g${OUTPUT_LEN}_tp${TP}.log"
        echo "Start process log file $LOG_FILE"
        if [ -f $LOG_FILE ]; then
            # Check if the log file contains "Throughput" (indicating a successful run)
            if grep -q "Throughput" $LOG_FILE; then
                COUNTER=0
                TOTAL_GPU_BLOCKS=$(grep "# GPU blocks:" "$LOG_FILE"|awk '{gsub(/,/, "", $(NF-4)); print $(NF-4)}')
                printf "0,${MODEL},${TOTAL_GPU_BLOCKS},0,0,0\n" >> $CSV_FILE
                grep "GPU KV cache usage" "$LOG_FILE"|awk -v model="$MODEL" -v total_gpu_blocks="$TOTAL_GPU_BLOCKS" -v counter=0 '
                {
                    gsub(/%,/, "", $(NF-5)); # Remove "%," from GPU KV cache
                    counter += 5; # Increment counter by 5 since the statistics interval are 5 seconds
                    print counter "," model "," total_gpu_blocks "," $(NF-17) "," $(NF-11) "," $(NF-5);
                }' >> $CSV_FILE
            else
                echo "ERROR: Log file $LOG_FILE is not complete"
            fi
        else
            echo "ERROR: Log file $LOG_FILE does not exist"
        fi
    done
}

##########################################################################################
for CACHE_TYPE in "${CACHE_TYPES[@]}"; do
    if [ "$CACHE_TYPE" = "percentage" ]; then
        process_percentage_evictfreq_logs $CACHE_TYPE
    elif [ "$CACHE_TYPE" = "full-cache" ]; then
        process_fullcache_logs $CACHE_TYPE
    else
        echo "Invalid cache type: $CACHE_TYPE"
        exit 1
    fi
done