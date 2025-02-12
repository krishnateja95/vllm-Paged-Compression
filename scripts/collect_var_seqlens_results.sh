#!/bin/bash
# p_latency: prompt latency; d_latency: decode latency; throughput: throughput
TARGET="p_latency"

# COMMON PARAMETERS
#INPUT_OUPUT_LENS=(1024 2048 4096 8192 16384 32768)
INPUT_OUPUT_LENS=(1024 2048 4096 8192)
NUM_REQ=50

### Models to test
# MODELS=("Llama-3.1-8B-Instruct" "Mistral-7B-Instruct-v0.2" "Qwen2.5-7B-Instruct-1M")
MODELS=("Llama-3.1-8B-Instruct")

BLOCK_SIZES=(16)

PROMPT_AND_DECODE_EVICT_METHODS=(
    "streamingLLM streamingLLM"
    # "streamingLLM value_l2"
    "value_l2 value_l2"
    # "value_l2 streamingLLM"
)

TP=1
BASE_LOG_DIR="$HOME/acl25/performance_logs/various_seqlens_tp${TP}/"
OUTPUT_DIR="$HOME/acl25/csv_results/"

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

################ Get Throughpt ########################################################
process_percentage_evictfreq_logs() {
    local EVICT_FREQS=(2 4)

    for MODEL in "${MODELS[@]}"; do 
        local CSV_FILE="${OUTPUT_DIR}/percentage_m${MODEL}.csv"
        if [ ! -f $CSV_FILE ]; then
            printf "model_name,evict_method,block_size,evict_freq,num_blocks_merge,input_size,output_size,num_reqs,throughput\n" > "$CSV_FILE"
        fi
        for INPUT_OUPUT_LEN in "${INPUT_OUPUT_LENS[@]}"; do
            INPUT_LEN=$INPUT_OUPUT_LEN
            OUTPUT_LEN=$INPUT_OUPUT_LEN
            echo "Processing the logs running with input/output length: $INPUT_LEN"   
            for P_AND_D_EVICT_METHOD in "${PROMPT_AND_DECODE_EVICT_METHODS[@]}"; do
                read -r P_EVICT_METHOD D_EVICT_METHOD <<< "$P_AND_D_EVICT_METHOD"
                EVICT_METHOD_DIR="${P_EVICT_METHOD}_${D_EVICT_METHOD}"
                EVICT_METHOD="${P_EVICT_METHOD}+${D_EVICT_METHOD}" 
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
    for MODEL in "${MODELS[@]}"; do
        local CSV_FILE="${OUTPUT_DIR}/fullcache_m${MODEL}.csv"
        if [ ! -f $CSV_FILE ]; then
            printf "model_name,evict_method,block_size,evict_size,input_size,output_size,num_reqs,throughput\n" > $CSV_FILE 
        fi
        for INPUT_OUPUT_LEN in "${INPUT_OUPUT_LENS[@]}"; do
            INPUT_LEN=$INPUT_OUPUT_LEN
            OUTPUT_LEN=$INPUT_OUPUT_LEN
            echo "Processing the logs running with input/output length: $INPUT_LEN"  
            for BLOCK_SIZE in "${BLOCK_SIZES[@]}"; do 
                LOG_FILE="${BASE_LOG_DIR}/full_cache/result_m${MODEL}_b${BLOCK_SIZE}_r${NUM_REQ}_p${INPUT_LEN}_g${OUTPUT_LEN}_tp${TP}.log"
                THROUGHPUT=$(get_throughput "$LOG_FILE")
                # Append the result to the CSV file
                printf "%s,%s,%s,%s,%s,%s,%s,%s\n" "$MODEL" "$CACHE_TYPE" "$BLOCK_SIZE" "0" "$INPUT_LEN" "$OUTPUT_LEN" "$NUM_REQ" "$THROUGHPUT" >> $CSV_FILE
            done
        done
    done  
}

################ Get Latency ########################################################
get_prompt_latency() {
    local LOG_FILE=$1
    local PRUNE_PROMPT_LATENCY="N/A"
    
    if [ -f $LOG_FILE ]; then
        # Parse the log file and extract the prompt prunning latency
        PRUNE_PROMPT_LATENCY=$(grep "Prune" $LOG_FILE | grep "prompt requests" | awk '{sum += $(NF-1)} END {print sum}')
    fi

    # Return the latency value
    echo $PRUNE_PROMPT_LATENCY
}

get_decode_latency() {
    #TODO: Change this function
    local LOG_FILE=$1
    local PRUNE_DECODE_LATENCY="N/A"
    
    if [ -f $LOG_FILE ]; then
        # Parse the log file and extract the prompt prunning latency
        PRUNE_DECODE_LATENCY=$(grep "Prune decode requests" $LOG_FILE | awk -F'total_dur=' '{sum += $2+0} END {print sum}')
    fi

    # Return the latency value
    echo $PRUNE_DECODE_LATENCY
}

process_prompt_latency_evictfreq_logs() {
    local EVICT_FREQS=(2 4)

    local CSV_FILE="${OUTPUT_DIR}/prompt_latency_r${NUM_REQ}.csv"
    if [ ! -f $CSV_FILE ]; then
        printf "model_name,evict_method,block_size,evict_freq,input_size,output_size,latency(s)\n" > "$CSV_FILE"
    fi
    CACHE_TYPE="percentage"
    for MODEL in "${MODELS[@]}"; do
        for INPUT_OUPUT_LEN in "${INPUT_OUPUT_LENS[@]}"; do
            INPUT_LEN=$INPUT_OUPUT_LEN
            OUTPUT_LEN=$INPUT_OUPUT_LEN
            echo "Processing the logs running with input/output length: $INPUT_LEN"
            for P_AND_D_EVICT_METHOD in "${PROMPT_AND_DECODE_EVICT_METHODS[@]}"; do
                read -r P_EVICT_METHOD D_EVICT_METHOD <<< "$P_AND_D_EVICT_METHOD"
                EVICT_METHOD_DIR="${P_EVICT_METHOD}_${D_EVICT_METHOD}"
                EVICT_METHOD="${P_EVICT_METHOD}+${D_EVICT_METHOD}" 
                for BLOCK_SIZE in "${BLOCK_SIZES[@]}"; do
                    for EVICT_FREQ in "${EVICT_FREQS[@]}"; do
                        LOG_FILE="${BASE_LOG_DIR}${CACHE_TYPE}/${EVICT_METHOD_DIR}/result_m${MODEL}_b${BLOCK_SIZE}_ef${EVICT_FREQ}_r${NUM_REQ}_p${INPUT_LEN}_g${OUTPUT_LEN}_tp${TP}.log"
                        PRUNE_PROMPT_LATENCY=$(get_prompt_latency "$LOG_FILE")
                        # Append the result to the CSV file
                        printf "%s,%s,%s,%s,%s,%s,%s\n" "$MODEL" "$EVICT_METHOD" "$BLOCK_SIZE" "$EVICT_FREQ" "$INPUT_LEN" "$OUTPUT_LEN" "$PRUNE_PROMPT_LATENCY" >> "$CSV_FILE"
                    done
                done
            done
        done
    done 
}

process_decode_latency_evictfreq_logs() {
    local EVICT_FREQS=(2 4)

    local CSV_FILE="${OUTPUT_DIR}/decode_latency_r${NUM_REQ}.csv"
    if [ ! -f $CSV_FILE ]; then
        printf "model_name,evict_method,block_size,evict_freq,input_size,output_size,latency(s)\n" > "$CSV_FILE"
    fi
    CACHE_TYPE="percentage"
    for MODEL in "${MODELS[@]}"; do
        for INPUT_OUPUT_LEN in "${INPUT_OUPUT_LENS[@]}"; do
            INPUT_LEN=$INPUT_OUPUT_LEN
            OUTPUT_LEN=$INPUT_OUPUT_LEN
            echo "Processing the logs running with input/output length: $INPUT_LEN"
            for P_AND_D_EVICT_METHOD in "${PROMPT_AND_DECODE_EVICT_METHODS[@]}"; do
                read -r P_EVICT_METHOD D_EVICT_METHOD <<< "$P_AND_D_EVICT_METHOD"
                EVICT_METHOD_DIR="${P_EVICT_METHOD}_${D_EVICT_METHOD}"
                EVICT_METHOD="${P_EVICT_METHOD}+${D_EVICT_METHOD}" 
                for BLOCK_SIZE in "${BLOCK_SIZES[@]}"; do
                    for EVICT_FREQ in "${EVICT_FREQS[@]}"; do
                        LOG_FILE="${BASE_LOG_DIR}${CACHE_TYPE}/${EVICT_METHOD_DIR}/result_m${MODEL}_b${BLOCK_SIZE}_ef${EVICT_FREQ}_r${NUM_REQ}_p${INPUT_LEN}_g${OUTPUT_LEN}_tp${TP}.log"
                        PRUNE_DECODE_LATENCY=$(get_decode_latency "$LOG_FILE")
                        # Append the result to the CSV file
                        printf "%s,%s,%s,%s,%s,%s,%s\n" "$MODEL" "$EVICT_METHOD" "$BLOCK_SIZE" "$EVICT_FREQ" "$INPUT_LEN" "$OUTPUT_LEN" "$PRUNE_DECODE_LATENCY" >> "$CSV_FILE"
                    done
                done
            done
        done
    done 
}

##########################################################################################
if [ $TARGET = "p_latency" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}/var_seqlens_tp${TP}_latency/"
    if [ ! -d "$OUTPUT_DIR" ]; then
        mkdir -p "$OUTPUT_DIR"
    fi
    process_prompt_latency_evictfreq_logs
elif [ $TARGET = "d_latency" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}/var_seqlens_tp${TP}_latency/"
    process_decode_latency_evictfreq_logs
elif [ $TARGET = "throughput" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}/var_seqlens_tp${TP}_thrpt/"
    if [ ! -d "$OUTPUT_DIR" ]; then
        mkdir -p "$OUTPUT_DIR"
    fi

    CACHE_TYPES=("percentage" "full-cache") 
    for CACHE_TYPE in "${CACHE_TYPES[@]}"; do
        if [ "$CACHE_TYPE" = "percentage" ]; then
            process_percentage_evictfreq_logs $CACHE_TYPE
        elif [ "$CACHE_TYPE" = "full-cache" ]; then
            process_fullcache_logs $CACHE_TYPE
        fi
    done
else
    echo "Invalid target: $TARGET"
    exit 1
fi
