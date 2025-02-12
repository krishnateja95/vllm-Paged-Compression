#!/bin/bash
# COMMON PARAMETERS
INPUT_LEN=1024
OUTPUT_LEN=8192
NUM_REQ=30

BLOCK_SIZE=16

# CACHE_TYPE="full-cache"
CACHE_TYPES=("full-cache" "percentage")

### Models to test
MODELS=("Llama-3.1-8B-Instruct" "Mistral-7B-Instruct-v0.2" "Qwen2.5-7B-Instruct-1M")
# MODELS=("Llama-3.1-8B-Instruct")

PROMPT_AND_DECODE_EVICT_METHODS=(
    "streamingLLM streamingLLM"
    "streamingLLM value_l2"
    "value_l2 value_l2"
    "value_l2 streamingLLM"
)

TP=1
BASE_LOG_DIR="$HOME/acl25/performance_logs/various_reqs_tp${TP}/"
OUTPUT_DIR="$HOME/acl25/csv_results/var_reqs_tp${TP}/"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

#####################################################################################
# Helper functions
# Get the number of recomputation triggered during inference
get_num_recomputes() {
    local LOG_FILE=$1
    local NUM_RERECOMPUTE="N/A"
    
    if [ -f $LOG_FILE ]; then
        # Check if the log file contains "Throughput" (indicating a successful run)
        if grep -q "Throughput" $LOG_FILE; then
            # Parse the log file and extract the number of recomputes
            NUM_RERECOMPUTE=$(grep "Scheduler preempting" "$LOG_FILE" | grep "RECOMPUTE" | wc -l)
        fi
    fi
    # Return the NUM_RERECOMPUTE value
    echo $NUM_RERECOMPUTE
}

process_percentage_evictfreq_logs() {
    CACHE_TYPE=$1
    local EVICT_FREQ=2

    for P_AND_D_EVICT_METHOD in "${PROMPT_AND_DECODE_EVICT_METHODS[@]}"; do
        read -r P_EVICT_METHOD D_EVICT_METHOD <<< "$P_AND_D_EVICT_METHOD"
        EVICT_METHOD_DIR="${P_EVICT_METHOD}_${D_EVICT_METHOD}"
        EVICT_METHOD="${P_EVICT_METHOD}+${D_EVICT_METHOD}" 
        for MODEL in "${MODELS[@]}"; do
            LOG_FILE="${BASE_LOG_DIR}${CACHE_TYPE}/${EVICT_METHOD_DIR}/result_m${MODEL}_b${BLOCK_SIZE}_ef${EVICT_FREQ}_r${NUM_REQ}_p${INPUT_LEN}_g${OUTPUT_LEN}_tp${TP}.log"
            echo "Start process log file $LOG_FILE"
            NUM_RERECOMPUTE=$(get_num_recomputes "$LOG_FILE")
            # Append the result to the CSV file
            printf "%s,%s,%s\n" "$MODEL" "$EVICT_METHOD" "$NUM_RERECOMPUTE" >> "$CSV_FILE"
        done
    done 
}

process_fullcache_logs() {
    CACHE_TYPE=$1
    for MODEL in "${MODELS[@]}"; do
        LOG_FILE="${BASE_LOG_DIR}${CACHE_TYPE}/result_m${MODEL}_b${BLOCK_SIZE}_r${NUM_REQ}_p${INPUT_LEN}_g${OUTPUT_LEN}_tp${TP}.log"
        echo "Start process log file $LOG_FILE"
        NUM_RERECOMPUTE=$(get_num_recomputes "$LOG_FILE")
        # Append the result to the CSV file
        printf "%s,%s,%s\n" "$MODEL" "$CACHE_TYPE" "$NUM_RERECOMPUTE" >> $CSV_FILE
    done
}

##########################################################################################
CSV_FILE="${OUTPUT_DIR}num_recomputes_p${INPUT_LEN}_g${OUTPUT_LEN}_r${NUM_REQ}_b${BLOCK_SIZE}.csv"
if [ ! -f $CSV_FILE ]; then 
    printf "model_name,evict_method,num_recompute\n" > $CSV_FILE 
fi

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