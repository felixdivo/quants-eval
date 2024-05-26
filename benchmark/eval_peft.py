import json
import os
import sys
import re

import datasets
import evaluate
import torch
import torch.distributed as dist
from accelerate import PartialState
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

from utils.arg_utils import (
    DataTrainingArguments,
    ModelArguments,
    QLoRaTrainingArguments,
    parse_hf_args,
)
from utils.data_utils import (
    CausalDataPreprocess,
    extract_answers,
    load_data_func,
    stratified_samples,
)

datasets.disable_progress_bars()
dist.init_process_group()
local_rank = dist.get_rank()


def get_answer(preds, prefix="### Answer: ", suffix=""):
    ret = []
    for pred in preds:
        try:
            idx = pred.index(prefix)
        except:  # noqa: E722
            ret.append("")
            continue

        if len(suffix) == 0:
            ret.append(pred[idx + len(prefix) :])
        else:
            ret.append(pred[idx + len(prefix) : -len(suffix)])
        if ret[-1].endswith("\n"):
            ret[-1] = ret[-1][:-1]
        if len(ret[-1]) > 2 and ret[-1][:2] in ["A:", "B:", "C:"]:
            ret[-1] = ret[-1][2:].strip()
    return ret


def gen_dataset(preds, ids, pred=False) -> Dataset:
    def gen():
        for i, e in zip(ids, preds):
            if pred:
                yield {"id": i, "prediction_text": e}
            else:
                yield {"id": i, "answers": [{"text": e, "answer_start": 0}]}

    if pred:
        features = datasets.Features(
            {
                "id": datasets.Value(dtype="string"),
                "prediction_text": datasets.Value(dtype="string"),
            }
        )
    else:
        features = datasets.Features(
            {
                "id": datasets.Value(dtype="string"),
                "answers": datasets.Sequence(
                    feature={
                        "text": datasets.Value(dtype="string"),
                        "answer_start": datasets.Value(dtype="int32"),
                    }
                ),
            }
        )
    return Dataset.from_generator(gen, features=features)


def main(args):
    parser_classes = [
        DataTrainingArguments,
        ModelArguments,
        TrainingArguments,
        QLoRaTrainingArguments,
    ]
    data_args, model_args, training_args, qlora_args = parse_hf_args(
        args, parser_classes
    )

    # Get a list of all files in the current directory
    try:
        files = os.listdir(training_args.output_dir)
    except FileNotFoundError:
        print("Output dir is empty nor not existed. Exitting!")
        sys.exit()

    # Get the latest checkpoint
    ckpt_files = [f for f in files if re.match(r"checkpoint-\d+", f)]
    if not ckpt_files:
        raise FileNotFoundError("No checkpoint found.")

    ckpt_files.sort(key=lambda f: int(f.split("-")[1]))
    training_args.ckpt = f"{training_args.output_dir}/{ckpt_files[-1]}"
    model_name = model_args.model_name_or_path

    metric = evaluate.load("squad", cache_dir=model_args.cache_dir)

    prefix = "### Answer: "
    suffix = ""
    text_field = "msg"
    prep = CausalDataPreprocess(data_args, text_field)
    format_func_eval = prep.formatting_val_func

    dataset, _ = load_data_func(data_args, model_args.token)
    dataset = dataset.map(
        format_func_eval, load_from_cache_file=False, keep_in_memory=True
    )
    max_samples_list = [data_args.max_eval_samples, data_args.max_predict_samples]

    eval_set_name = ["test", "val"][local_rank]
    max_samples = max_samples_list[local_rank]
    eval_set = dataset[eval_set_name]
    if max_samples is not None:
        max_samples = min(len(eval_set), max_samples)
        eval_set = stratified_samples(eval_set, max_samples, "question_type", 0)

    # Load the GG model - this is the local one, update it to the one on the Hub
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        training_args.output_dir,
        quantization_config=quantization_config,
        torch_dtype=torch.float32,
        cache_dir=model_args.cache_dir,
        device_map={"": PartialState().process_index},
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        token=model_args.token,
        max_seq_length=data_args.max_seq_length,
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    if "falcon" in model_name:
        target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
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

    peft_model = get_peft_model(model, lora_config)
    peft_model = peft_model.from_pretrained(
        model, training_args.ckpt, config=lora_config
    )
    peft_model.eval()

    batch_size = training_args.per_device_eval_batch_size
    n_batch = len(eval_set) // batch_size
    preds = []
    answers = []
    ids = []

    for i in tqdm(
        range(n_batch), position=0, file=sys.stdout, miniters=500, mininterval=15
    ):
        if i == n_batch - 1:
            inputs = eval_set[text_field][i * batch_size :]
            ids += eval_set["id"][i * batch_size :]
        else:
            inputs = eval_set[text_field][i * batch_size : (i + 1) * batch_size]
            ids += eval_set["id"][i * batch_size : (i + 1) * batch_size]

        inputs = tokenizer(inputs, return_tensors="pt", padding=True).to("cuda")
        max_length = data_args.max_answer_length
        outputs = peft_model.generate(
            **inputs, max_new_tokens=max_length, pad_token_id=tokenizer.eos_token_id
        )
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        preds += get_answer(outputs, prefix, suffix)
    if data_args.answer_column is None:
        answers = extract_answers(eval_set)
    else:
        answers = eval_set[data_args.answer_column]
    refs = gen_dataset(answers, ids)

    preds = gen_dataset(preds, ids, True)

    results = metric.compute(predictions=preds, references=refs)
    max_train_samples = (
        "full" if data_args.max_train_samples else str(data_args.max_train_samples)
    )
    results["stage"] = eval_set_name
    results["train_size"] = max_train_samples

    if data_args.log_dir is not None:
        file_name = "_".join(
            [
                data_args.log_dir,
                max_train_samples,
                f"{eval_set_name}.csv",
            ]
        )
        preds.to_csv(file_name)
        file_name = f"{data_args.log_dir}_metrics.jsonl"
        with open(file_name, "a") as file:
            file.write(json.dumps(results) + " \n")

    print(results)


if __name__ == "__main__":
    main(sys.argv[1:])
