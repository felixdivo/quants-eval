# Benchmark code

For models that require QLoRA to train, run `finetune.py` first and then run `eval_peft.py` to do
evaluation(see `run_llm.sh` for an example). The `eval_peft.py` file is configured to run with a system
with 1 node and 2 gpus, modify it to fit your system.

The `run_small.sh` file supports evaluations after training but it only supports a small number of
models (see [HuggingFace Doc]([https://](https://huggingface.co/docs/transformers/v4.39.1/en/model_doc/auto#transformers.AutoModelForSeq2SeqLM)))
