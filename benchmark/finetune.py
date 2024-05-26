#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library's seq2seq models for question answering using the 🤗 Seq2SeqTrainer.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import json
import logging
import os
import sys

import datasets
import evaluate
import torch
import transformers
from accelerate import PartialState
from peft import LoraConfig
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import (
    EvalPrediction,
    get_last_checkpoint,
)
from trl import SFTTrainer

from trainer.seq2seq_trainer_qa import QASeq2SeqTrainer
from utils.arg_utils import (
    DataTrainingArguments,
    ModelArguments,
    QLoRaTrainingArguments,
    parse_hf_args,
)
from utils.data_utils import (
    CausalDataPreprocess,
    Seq2SeqDataPreprocess,
    load_data_func,
    stratified_samples,
)

# from transformers.utils import check_min_version, send_example_telemetry
# from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.40.0.dev0")

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

logger = logging.getLogger(__name__)
datasets.disable_progress_bars()


def main(args: list[str]):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    if "--qlora" in args:
        args.remove("--qlora")
        train_w_qlora = True
        parser_classes = [
            DataTrainingArguments,
            ModelArguments,
            TrainingArguments,
            QLoRaTrainingArguments,
        ]
        data_args, model_args, training_args, qlora_args = parse_hf_args(
            args, parser_classes
        )
        logger.info("Training with QLoRA.")
    else:
        train_w_qlora = False
        parser_classes = [
            DataTrainingArguments,
            ModelArguments,
            Seq2SeqTrainingArguments,
        ]
        data_args, model_args, training_args = parse_hf_args(args, parser_classes)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        token=model_args.token,
        max_seq_length=data_args.max_seq_length,
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
    )
    # Load metric
    metric = evaluate.load("squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Set seed before initializing model.
    set_seed(training_args.seed)
    raw_datasets, column_names = load_data_func(data_args, model_args.token)

    if train_w_qlora:
        data_preprocessor = CausalDataPreprocess(
            data_args, eos_token=tokenizer.eos_token
        )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        if "falcon" in model_args.model_name_or_path:
            target_modules = [
                "query_key_value",
                "dense",
                "dense_h_to_4h",
                "dense_4h_to_h",
            ]
        else:
            target_modules = [
                "q_proj",
                "o_proj",
                "k_proj",
                "v_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        lora_config = LoraConfig(
            r=qlora_args.lora_r,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
            lora_alpha=qlora_args.lora_alpha,
            lora_dropout=qlora_args.lora_dropout,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            torch_dtype=torch.float16,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            quantization_config=quantization_config,
            cache_dir=model_args.cache_dir,
            device_map={"": PartialState().process_index},
        )
    else:
        data_preprocessor = Seq2SeqDataPreprocess(tokenizer, data_args)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
        # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
        # on a small vocab and want a smaller embedding size, remove this test.
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))

        if model.config.decoder_start_token_id is None:
            raise ValueError(
                "Make sure that `config.decoder_start_token_id` is correctly defined"
            )

    # Temporarily set max_answer_length for training.
    if training_args.label_smoothing_factor > 0 and not hasattr(
        model, "prepare_decoder_input_ids_from_labels"
    ):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for "
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"].shuffle(seed=0)
        if data_args.max_train_samples is not None:
            # We will select sample from whole data if argument is specified
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = stratified_samples(
                train_dataset,
                max_train_samples,
                col="question_type",
                seed=0,
            )
        # Create train feature from dataset
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            if not train_w_qlora:
                train_dataset = train_dataset.map(
                    data_preprocessor.preprocess_func,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on train dataset",
                )
        if data_args.max_train_samples is not None:
            # Number of samples might increase during Feature Creation, We select only specified max samples
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval and not train_w_qlora:
        if "val" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["val"]
        if data_args.max_eval_samples is not None:
            # We will select sample from whole data
            max_eval_samples = min(len(eval_examples), data_args.max_eval_samples)
            eval_examples = eval_examples.select(range(max_eval_samples))
        # Validation Feature Creation
        with training_args.main_process_first(
            desc="validation dataset map pre-processing"
        ):
            eval_dataset = eval_examples.map(
                data_preprocessor.preprocess_val_func,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
                keep_in_memory=True,
                batched=True,
            )
        if data_args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict and not train_w_qlora:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_examples = predict_examples.select(
                range(data_args.max_predict_samples)
            )
        # Predict Feature Creation
        with training_args.main_process_first(
            desc="prediction dataset map pre-processing"
        ):
            predict_dataset = predict_examples.map(
                data_preprocessor.preprocess_val_func,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
                keep_in_memory=True,
                batched=True,
            )
        if data_args.max_predict_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples
            )
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    if train_w_qlora:
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None and tokenizer.bos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.bos_token_id
        elif tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.additional_special_tokens_ids[0]

        # Initialize Trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=train_dataset if training_args.do_train else None,
            peft_config=lora_config,
            max_seq_length=data_args.max_seq_length,
            formatting_func=data_preprocessor.formatting_func,
        )
    else:
        label_pad_token_id = (
            -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        )
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

        # Initialize our Trainer
        training_args.find_unused_parameters = False
        trainer = QASeq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            eval_examples=eval_examples if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
            if training_args.predict_with_generate
            else None,
            answer_column=data_args.answer_column,
        )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    max_train_samples = (
        "full" if data_args.max_train_samples else str(data_args.max_train_samples)
    )
    results = {}
    max_length = data_args.max_answer_length
    if training_args.do_eval and not train_w_qlora:
        logger.info("*** Validation ***")
        file = None
        if data_args.log_dir is not None:
            file_name = "_".join(
                [
                    data_args.log_dir,
                    max_train_samples,
                    "val.csv",
                ]
            )
            file = open(file_name, "a")
        metrics = trainer.evaluate(
            max_new_tokens=max_length, metric_key_prefix="val", output_file=file
        )

        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        if data_args.log_dir is not None:
            file.close()
            file_name = f"{data_args.log_dir}_metrics.jsonl"
            with open(file_name, "a") as file:
                metrics["stage"] = "val"
                metrics["train_size"] = max_train_samples
                file.write(str(metrics) + "\n")

        trainer.log_metrics("val", metrics)
        trainer.save_metrics("val", metrics)

    # Prediction
    if training_args.do_predict and not train_w_qlora:
        logger.info("*** Test ***")
        file = None
        if data_args.log_dir is not None:
            file_name = "_".join(
                [
                    data_args.log_dir,
                    max_train_samples,
                    "test.csv",
                ]
            )
            file = open(file_name, "a")
        results = trainer.predict(
            predict_dataset,
            predict_examples,
            max_new_tokens=max_length,
            output_file=file,
        )

        metrics = results.metrics

        max_predict_samples = (
            data_args.max_predict_samples
            if data_args.max_predict_samples is not None
            else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        if data_args.log_dir is not None:
            file.close()
            file_name = f"{data_args.log_dir}_metrics.jsonl"
            with open(file_name, "a") as file:
                metrics["stage"] = "test"
                metrics["train_size"] = max_train_samples
                file.write(json.dumps(metrics) + "\n")

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main(sys.argv[1:])
