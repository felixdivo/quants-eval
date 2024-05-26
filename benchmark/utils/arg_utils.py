import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser


MODEL_NAME_MAPPING = {
    "t5_small": "google-t5/t5-small",
    "t5_base": "google-t5/t5-base",
    "mvp": "RUCAIBox/mvp",
    "umt5_small": "google/umt5-small",
    "umt5_base": "google/umt5-base",
    "gemma_2b": "google/gemma-2b",
    "gemma_7b": "google/gemma-7b",
    "llama2": "meta-llama/llama-2-7b-hf",
    "mistral": "mistralai/mistral-7B-v0.1",
    "falcon": "tiiuae/falcon-7b",
}


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default="/scratch/wn86/an9349/models/",
        metadata={
            "help": "Path to directory to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )


@dataclass
class QLoRaTrainingArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=8)
    use_flash_attention_2: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash Attention 2."},
    )
    padding_size: Optional[str] = field(default="left")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_task: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    context_column: str = field(
        default="context",
        metadata={
            "help": "The name of the column in the datasets containing the contexts (for question answering)."
        },
    )
    question_column: str = field(
        default="question",
        metadata={
            "help": "The name of the column in the datasets containing the questions (for question answering)."
        },
    )
    answer_column: str = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the answers (for question answering)."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the perplexity on (a text file)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=256,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_answer_length: int = field(
        default=16,
        metadata={
            "help": (
                "The maximum length of an answer that can be generated. This is needed because the start "
                "and end predictions are not conditioned on one another."
            )
        },
    )
    val_max_answer_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_answer_length`. "
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when"
                " batching to the maximum length in the batch (which can be faster on GPU but will be slower on TPU)."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    log_dir: Optional[str] = field(
        default=None,
        metadata={"help": ("Path to log directory.")},
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file/test_file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`test_file` should be a csv or a json file."
        if self.val_max_answer_length is None:
            self.val_max_answer_length = self.max_answer_length

        if self.log_dir is not None and not self.log_dir.endswith("/"):
            self.log_dir += "/"


def parse_hf_args(args, parser_classes):
    parser = HfArgumentParser(parser_classes)
    args = parser.parse_args_into_dataclasses(args)

    # args = [data_args, model_args, training_args, [qlora_args]]
    if args[2].output_dir[0] == "$":
        end_idx = args[2].output_dir.index("/")
        env_var = args[2].output_dir[1:end_idx]
        args[2].output_dir = args[2].output_dir.replace(
            f"${env_var}", os.environ[env_var]
        )

    if args[1].model_name_or_path in MODEL_NAME_MAPPING:
        model_name_short = args[1].model_name_or_path
        args[1].model_name_or_path = MODEL_NAME_MAPPING[model_name_short]
    else:
        model_name_short = (
            args[1]
            .model_name_or_path.split("/")[-1]
            .replace("-", "_")
            .replace(".", "_")
            .lower()
        )
    args[2].output_dir += (
        model_name_short if args[2].output_dir[-1] == "/" else f"/{model_name_short}"
    )
    if args[0].log_dir is not None:
        args[0].log_dir += model_name_short
    return args
