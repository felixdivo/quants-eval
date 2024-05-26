#!/bin/bash

# Model aliases
# "t5_small": "google-t5/t5-small",
# "t5_base": "google-t5/t5-base",
# "mvp": "RUCAIBox/mvp",
# "umt5_small": "google/umt5-small",
# "umt5_base": "google/umt5-base",
# "gemma_2b": "google/gemma-2b",
# "gemma_7b": "google/gemma-7b",
# "llama2": "meta-llama/llama-2-7b-hf",
# "mistral": "mistralai/mistral-7B-v0.1",
# "falcon": "tiiuae/falcon-7b"

MODEL=gemma_2b
OUTPUT_DIR=$HOME/output/
LOG_DIR=$HOME/logs/

TRAIN_SAMPLES=( $(seq 50 50 500) )
LEARNING_RATES=(0.001 0.001 0.0005 0.0005 0.0005 0.0002 0.0002 0.0001 0.0001 0.0001)

# Get the length of the arrays
LENGTH=${#TRAIN_SAMPLES[@]}

# Iterate over the arrays
for (( i=0; i<$LENGTH; i++ ))
do
    TRAIN_SAMPLE=${TRAIN_SAMPLES[$i]}
    LEARNING_RATE=${LEARNING_RATES[$i]}

    # rm -r $OUTPUT_DIR # Clear the output directory

    torchrun --nproc_per_node 2 ~/finetune.py \
        --qlora \
        --model_name_or_path $MODEL \
        --output_dir $OUTPUT_DIR \
        --log_dir $LOG_DIR \
        --dataset_name dasyd/time-qa \
        --context_column textual_description \
        --question_column question \
        --do_train \
        --do_eval \
        --lora_alpha 32 \
        --lora_dropout 0.1 \
        --lora_r 16 \
        --fp16 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --optim paged_adamw_32bit \
        --num_train_epochs 5 \
        --save_strategy epoch \
        --logging_strategy epoch \
        --save_total_limit 1 \
        --max_seq_length 256 \
        --max_answer_length 16 \
        --learning_rate $LEARNING_RATE \
        --max_train_samples $TRAIN_SAMPLE

    torchrun --nproc_per_node 2 ~/eval_peft.py \
        --model_name_or_path $MODEL \
        --output_dir $OUTPUT_DIR \
        --log_dir $LOG_DIR \
        --dataset_name dasyd/time-qa \
        --context_column textual_description \
        --question_column question \
        --lora_alpha 32 \
        --lora_dropout 0.1 \
        --lora_r 16 \
        --fp16 true \
        --per_device_eval_batch_size 8 \
        --max_seq_length 256 \
        --max_answer_length 16 \
        --max_train_samples $TRAIN_SAMPLE

done
