#!/bin/bash
CACHE_TYPE="full-cache"
# CACHE_TYPE="percentage"

# COMMON PARAMETERS
INPUT_LEN=1024
OUTPUT_LEN=8192
NUM_REQS=(10 20 30 40 50 60 70 80 90 100)

### Models to test
MODELS=("Llama-3.1-8B-Instruct" "Mistral-7B-Instruct-v0.2")

BLOCK_SIZES(8 16 32)

PROMPT_AND_DECODE_EVICT_METHODS=(
    "streamingLLM streamingLLM"
    # "streamingLLM value_l2"
    "value_l2 value_l2"
    # "value_l2 streamingLLM"
)

TP=1
BASE_LOG_DIR="$HOME/acl25/performance_logs/"
OUTPUT_DIR="$HOME/acl25/csv_results/var_reqs_tp${TP}/"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

#####################################################################################
# Helper functions
get_throughput() {
    local LOG_FILE=$1
    local THROUGHPUT
    
    if [ -f $LOG_FILE ]; then
        # Parse the log file and extract the throughput
        THROUGHPUT=$(grep "Throughput" $LOG_FILE | cut -d ' ' -f 4)
        if [ -z "$THROUGHPUT" ]; then
            THROUGHPUT="N/A"
        fi
    else
        THROUGHPUT="N/A"  # Default value if file is missing
    fi

    # Return the throughput value
    echo $THROUGHPUT
}

process_percentage_evictfreq_logs() {
    local EVICT_FREQS=(2 4)

    local CSV_FILE="${OUTPUT_DIR}/percentage_p${INPUT_LEN}_g${OUTPUT_LEN}_tp${TP}.csv"
    if [ -f $CSV_FILE ]; then
        printf "model_name,evict_method,block_size,evict_freq,num_blocks_merge,input_size,output_size,num_reqs,throughput\n" > "$CSV_FILE"
    fi 
    for NUM_REQ in "${NUM_REQS[@]}"; do
        for P_AND_D_EVICT_METHOD in "${PROMPT_AND_DECODE_EVICT_METHODS[@]}"; do
            read -r P_EVICT_METHOD D_EVICT_METHOD <<< "$P_AND_D_EVICT_METHOD"
            EVICT_METHOD_DIR="${P_EVICT_METHOD}_${D_EVICT_METHOD}"
            EVICT_METHOD="${P_EVICT_METHOD}+${D_EVICT_METHOD}" 
            for MODEL in "${MODELS[@]}"; do
                for BLOCK_SIZE in "${BLOCK_SIZES[@]}"; do
                    for EVICT_FREQ in "${EVICT_FREQS[@]}"; do
                        LOG_FILE="${BASE_LOG_DIR}${CACHE_TYPE}/${EVICT_METHOD_DIR}/result_m${MODEL}_b${BLOCK_SIZE}_ef${EVICT_FREQ}_r${NUM_REQ}_p${INPUT_LEN}_g${OUTPUT_LEN}_tp${TP}.log"
                        THROUGHPUT=$(get_throughput "$LOG_FILE")
                        # Append the result to the CSV file
                        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" "$MODEL" "$EVICT_METHOD" "$BLOCK_SIZE" "$EVICT_FREQ" "2" "$INPUT_LEN" "$OUTPUT_LEN" "$NUM_REQ" "$THROUGHPUT" >> "$CSV_FILE"
                    done
                done
            done
        done
    done 
}

process_fullcache_logs() {
    local CSV_FILE="${OUTPUT_DIR}/fullcache_p${INPUT_LEN}_g${OUTPUT_LEN}.csv"
    if [ -f $CSV_FILE ]; then 
        printf "model_name,evict_method,block_size,evict_freq,num_blocks_merge,input_size,output_size,num_reqs,throughput\n" > $CSV_FILE 
    fi
    for NUM_REQ in "${NUM_REQS[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            for BLOCK_SIZE in "${BLOCK_SIZES[@]}"; do
                LOG_FILE="${BASE_LOG_DIR}/full_cache/result_m${MODEL}_b${BLOCK_SIZE}_r${NUM_REQ}_p${INPUT_LEN}_g${OUTPUT_LEN}_tp${TP}.log"
                THROUGHPUT=$(get_throughput "$LOG_FILE")
                # Append the result to the CSV file
                printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" "$MODEL" "$CACHE_TYPE" "$BLOCK_SIZE" "-1" "-1" "$INPUT_LEN" "$OUTPUT_LEN" "$NUM_REQ" "$THROUGHPUT" >> $CSV_FILE
            done
        done
    done  
}

##########################################################################################

if [ "$CACHE_TYPE" = "percentage" ]; then
    # Define paired BLOCK_SIZE and EVICT_SIZE for the percentage cache type "BLOCK_SIZE EVICT_SIZE"
    process_percentage_evictfreq_logs
elif [ "$CACHE_TYPE" = "full-cache" ]; then
    process_fullcache_logs
else
    echo "Invalid cache type: $CACHE_TYPE"
    exit 1
fi