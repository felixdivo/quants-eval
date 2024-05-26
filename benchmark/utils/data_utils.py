from typing import List, Tuple

from datasets import load_dataset, Dataset
import random


def extract_answers(examples):
    true = "true"  # noqa
    false = "false"  # noqa
    q_types = examples["question_type"]
    answers = examples["correct_option"]
    if not q_types.endswith("open"):
        options = eval(examples["options"])
        answers = str(options[answers])
    return {"answer": answers}


def stratified_samples(dataset: Dataset, size: int, col: str, seed=0) -> Dataset:
    try:
        dataset_ = dataset.class_encode_column(col)
        dataset_ = dataset_.train_test_split(
            train_size=size, stratify_by_column=col, seed=seed
        )["train"]
        ids = dataset_["id"]
        return dataset.filter(lambda x: x["id"] in ids)
    except:  # noqa
        random.seed(seed)
        selected_samples = list(range(len(dataset)))
        selected_samples = random.choices(selected_samples, k=size)
        return dataset.select(selected_samples)


def load_data_func(data_args, token=None, use_trajectory=False, add_id_col=True):
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_task, token=token
        )
    else:
        # Load data from local files
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["val"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    column_names = list(raw_datasets.column_names.values())[0]

    if add_id_col:
        raw_datasets = raw_datasets.map(
            lambda example: {
                "id": f"{example['sample_id']:08}{example['question_id']}"
            },
            batched=False,
        )
        if data_args.answer_column is None:
            raw_datasets = raw_datasets.map(extract_answers)
            data_args.answer_column = "answer"
    if not use_trajectory and "trajectory" in column_names:
        raw_datasets = raw_datasets.remove_columns(["trajectory"])
    column_names = list(raw_datasets.column_names.values())[0]
    return raw_datasets, column_names


class CausalDataPreprocess:
    def __init__(self, data_args=None, text_field=None, eos_token=None) -> None:
        self.question_column = data_args.question_column
        self.context_column = data_args.context_column
        self.answer_column = data_args.answer_column
        self.text_field = text_field
        self.template = (
            "### Context: {c}\n\n### Question: {q}\n\n### Answer: {a}" + eos_token
            if eos_token is not None
            else ""
        )
        self.template_val = "### Context: {c}\n\n### Question: {q}\n\n"

    def formatting_func(self, example):
        context = example[self.context_column]
        question = example[self.question_column]
        answer = example[self.answer_column]

        if not isinstance(context, str):
            prompt = [
                self.template.format(c=c, q=q, a=a)
                for c, q, a in zip(context, question, answer)
            ]
            return prompt if self.text_field is None else {self.text_field: prompt}

        prompt = self.template.format(c=context, q=question, a=answer)
        if self.text_field is None:
            return prompt
        return [prompt] if self.text_field is None else {self.text_field: prompt}

    def formatting_val_func(self, example):
        context = example[self.context_column]
        question = example[self.question_column]

        if not isinstance(context, str):
            prompt = [
                self.template_val.format(c=c, q=q) for c, q in zip(context, question)
            ]
            return prompt if self.text_field is None else {self.text_field: prompt}

        prompt = self.template_val.format(c=context, q=question)
        return [prompt] if self.text_field is None else {self.text_field: prompt}


class Seq2SeqDataPreprocess:
    def __init__(self, tokenizer, data_args) -> None:
        self.tokenizer = tokenizer
        self.ignore_pad_token = data_args.ignore_pad_token_for_loss
        self.max_seq_len = min(data_args.max_seq_length, tokenizer.model_max_length)
        self.max_answer_len = data_args.max_answer_length
        self.padding = "max_length" if data_args.pad_to_max_length else False
        self.question_column = data_args.question_column
        self.context_column = data_args.context_column
        self.answer_column = data_args.answer_column

    def preprocess_batch(self, examples) -> Tuple[List[str], List[str]]:
        questions = examples[self.question_column]
        contexts = examples[self.context_column]
        targets = examples[self.answer_column]

        def generate_input(_question, _context):
            return " ".join(
                ["Question:", _question.lstrip(), "Context:", _context.lstrip()]
            )

        inputs = [
            generate_input(question, context)
            for question, context in zip(questions, contexts)
        ]
        return inputs, targets

    def preprocess_func(self, examples):
        inputs, targets = self.preprocess_batch(examples)

        model_inputs = self.tokenizer(
            inputs, max_length=self.max_seq_len, padding=self.padding, truncation=True
        )
        # Tokenize targets with text_target=...
        labels = self.tokenizer(
            text_target=targets,
            max_length=self.max_seq_len,
            padding=self.padding,
            truncation=True,
        )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if self.padding == "max_length" and self.ignore_pad_token:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label]  # noqa: E741
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Validation preprocessing
    def preprocess_val_func(self, examples):
        inputs, targets = self.preprocess_batch(examples)

        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_seq_len,
            padding=self.padding,
            truncation=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        # Tokenize targets with the `text_target` keyword argument
        labels = self.tokenizer(
            text_target=targets,
            max_length=self.max_answer_len,
            padding=self.padding,
            truncation=True,
        )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if self.padding == "max_length" and self.ignore_pad_token:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label]  # noqa: E741
                for label in labels["input_ids"]
            ]

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = model_inputs.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        model_inputs["example_id"] = []
        # Augment the overflowing tokens to the labels
        labels_out = []

        for i in range(len(model_inputs["input_ids"])):
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            model_inputs["example_id"].append(examples["id"][sample_index])
            labels_out.append(labels["input_ids"][sample_index])

        model_inputs["labels"] = labels_out
        return model_inputs
