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
LEARNING_RATE=1e-5
OUTPUT_DIR=$HOME/output/
LOG_DIR=$HOME/logs/

# rm -r $PBS_JOBFS/*
torchrun --nproc_per_node 2 ~/finetune.py \
    --model_name_or_path $MODEL \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR \
    --dataset_name dasyd/time-qa \
    --context_column textual_description \
    --question_column question \
    --do_train \
    --do_eval \
    --do_predict \
    --num_train_epochs 5 \
    --learning_rate $LEARNING_RATE \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --predict_with_generate true \
    --max_seq_length 256 \
    --max_answer_length 16 \
    --generation_max_length 16 \
    --max_train_samples 200
