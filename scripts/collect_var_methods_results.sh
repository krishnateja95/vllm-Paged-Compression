#!/bin/bash
EVICT_METHODS=("full-cache" "inverse_key_l2" "streamingLLM-1" "local" "global")

# COMMON PARAMETERS
INPUT_LEN=1024
OUTPUT_LEN=8192
# NUM_REQS=(64 128)
NUM_REQS=(64)
CACHE_BUDGETS=(256 512 1024 2048 4096)
TOPK_BLOCKS=(3)
FLAGS=('w_prefill_evict' 'wo_prefill_evict')

### Models to test
MODELS=("Llama-3.2-1B-Instruct" "Llama-3.2-3B-Instruct" "Llama-3.1-8B-Instruct")

BASE_LOG_DIR="$HOME/kv_evict/perf_logs/"
OUTPUT_DIR="$HOME/kv_evict/csv_results/"

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

process_evict_logs() {
    local evict_method=$1
    local base_log_path=$2
    local CSV_FILE="${OUTPUT_DIR}/evict_p${INPUT_LEN}_g${OUTPUT_LEN}.csv"
    # check of the CSV file exists, if not create it
    if [ ! -f "$CSV_FILE" ]; then
        # Create the CSV file and add the header
        printf "model_name,evict_method,input_size,output_size,num_reqs,cache_budget,enable_prefill_evict,throughput(tokens/s)\n" > "$CSV_FILE"
    fi

    for flag in "${FLAGS[@]}"; do
        enable_prefill_evict=1
        if [ "$flag" = "w_prefill_evict" ]; then
            base_log_path="${base_log_path}/enable_prefill_evict/${evict_method}" 
            enable_prefill_evict=1
        else
            base_log_path="${base_log_path}/disable_prefill_evict/${evict_method}"
            enable_prefill_evict=0
        fi
        
        if [ $evict_method = "local" ]; then
            for MODEL in "${MODELS[@]}"; do
                for TOP_K in "${TOPK_BLOCKS[@]}"; do
                    for NUM_REQ in "${NUM_REQS[@]}"; do
                        for BUDGET in "${CACHE_BUDGETS[@]}";do
                            # Define the block size and eviction frequency based on the method
                            LOG_FILE="${base_log_path}/${MODEL}_bd${BUDGET}_r${NUM_REQ}_p${INPUT_LEN}_g${OUTPUT_LEN}.log"
                            THROUGHPUT=$(get_throughput "$LOG_FILE")
                            # Append the result to the CSV file
                            printf "%s,%s,%s,%s,%s,%s,%s,%s\n" "$MODEL" "$evict_method-topk=${TOP_K}" "$INPUT_LEN" "$OUTPUT_LEN" "$NUM_REQ" "$BUDGET" "$enable_prefill_evict" "$THROUGHPUT" >> "$CSV_FILE"
                        done
                    done
                done
            done
        else
            for MODEL in "${MODELS[@]}"; do
                for NUM_REQ in "${NUM_REQS[@]}"; do
                    for BUDGET in "${CACHE_BUDGETS[@]}";do
                        # Define the block size and eviction frequency based on the method
                        LOG_FILE="${base_log_path}/${MODEL}_bd${BUDGET}_r${NUM_REQ}_p${INPUT_LEN}_g${OUTPUT_LEN}.log"
                        THROUGHPUT=$(get_throughput "$LOG_FILE")
                        # Append the result to the CSV file
                        printf "%s,%s,%s,%s,%s,%s,%s,%s\n" "$MODEL" "$evict_method" "$INPUT_LEN" "$OUTPUT_LEN" "$NUM_REQ" "$BUDGET" "$enable_prefill_evict" "$THROUGHPUT" >> "$CSV_FILE"
                    done
                done
            done
        fi 
    done 
}

process_fullcache_logs() {
    local base_log_path=$1
    local CSV_FILE="${OUTPUT_DIR}/fullcache_p${INPUT_LEN}_g${OUTPUT_LEN}.csv"
    if [ -f $CSV_FILE ]; then 
        printf "model_name,evict_method,input_size,output_size,num_reqs,throughput(tokens/s)\n" > $CSV_FILE 
    fi
    for NUM_REQ in "${NUM_REQS[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            LOG_FILE="${base_log_path}/${MODEL}_r${NUM_REQ}_p${INPUT_LEN}_g${OUTPUT_LEN}.log"
            THROUGHPUT=$(get_throughput "$LOG_FILE")
            # Append the result to the CSV file
            printf "%s,%s,%s,%s,%s,%s\n" "$MODEL" "full-cache" "$INPUT_LEN" "$OUTPUT_LEN" "$NUM_REQ" "$THROUGHPUT" >> $CSV_FILE
        done
    done  
}

##########################################################################################
for evict_method in "${EVICT_METHODS[@]}"; do
    if [ "$evict_method" = "default" ]; then
        base_log_path="${BASE_LOG_DIR}/full_cache/"
        process_fullcache_logs "$base_log_path"
    else
        base_log_path="${BASE_LOG_DIR}"
        process_evict_logs "$evict_method" "$base_log_path"
    fi
done
