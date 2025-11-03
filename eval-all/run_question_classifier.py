"""
See `run_llm_judge.py` for comments on how to start the LLM server.
"""

from time import sleep
from typing import Any, Literal
from pathlib import Path
import json
from collections import defaultdict
import shutil
import warnings
from argparse import ArgumentParser

import openai
import pandas as pd

from pydantic import BaseModel, Field
from openai import OpenAI
from sglang import set_default_backend, RuntimeEndpoint
from rich.progress import (
    Progress,
    MofNCompleteColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from concurrent.futures import ThreadPoolExecutor


def run_classifier(row: pd.Series, client: OpenAI) -> str:
    """Run the LLM judge on a single row of the evaluation dataset.

    Args:
        row: A row from the evaluation dataset containing the scene description, question, reference answer,
             and system answer.
        client: An instance of the OpenAI client to interact with the LLM judge.

    Returns:
        str: The JSON response from the LLM judge containing the rating and rationale.
    """

    PROMPT_TEMPLATE = """You are given a question and an answer. Please classify the type of question into one of the defined categories. Also, state the type of possible answer.

# The Question and Answer
Question: "{question}"
Answer: "{answer}"

# Possible Question Types
after: Did some specific event occur after some specific other one?
before: Did some particular event occur before another one?
comparison: Is the same event occurring before and after a certain one? (only binary)
comparison_counting: Do events A and B occur the same number of times? (only binary)
comparison_first_last_different: Are the first and last events different? (only binary)
comparison_first_last_same: Are the first and last events the same? (only binary)
comparison_timestamp_different: Are the actions at time point A and time point B different? (only binary)
comparison_timestamp_same: Are the actions at time point A and time point B the same? (only binary)
count: How many times does a particular event occur?
count_number: Which event occurs a specific number of times? (only multi and open)
descriptive_identification: At a certain time point, what is the event that occurs?
extremum_least: Was a particular event the least frequently occurring one? (only binary and open)
extremum_most: Was a particular event the most frequently occurring one? (only binary and open)
first: What is the very first event that occurs?
last: What is the very last event that occurs?
interval_part_sequence: How many different events occur in the interval between two specific time points?
interval_whole_sequence: How many different events occur in the entire sequence?
right_after: Does event A directly follow event B?
right_before: Does event A directly precede event B?
other: Any question that cannot be reasonably sorted into the above categories.

# Possible Answer Types
binary: The answer can either be yes/true or no/false.
multi: The answer is chosen from a set of multiple-choice options, like A/B/C.
open: A free-form type of response is expected.

# Response Format
Provide your assessment in the following JSON format:

{{
  "QuestionType": "<the most appropriate question type from above>",
  "AnswerType": "<binary/multi/open>"
}}

You MUST provide values for "QuestionType" and "AnswerType" in your answer. You MUST choose only from the provided options.
Do NOT respond with any other text in your answer. Immediately, start with the JSON object without additional text or explanation. Make sure to output complete and valid JSON."""

    class ClassifierResponse(BaseModel):
        QuestionType: Literal[
            "after",
            "before",
            "comparison",
            "comparison_counting",
            "comparison_first_last_different",
            "comparison_first_last_same",
            "comparison_timestamp_different",
            "comparison_timestamp_same",
            "count",
            "count_number",
            "descriptive_identification",
            "extremum_least",
            "extremum_most",
            "first",
            "last",
            "interval_part_sequence",
            "interval_whole_sequence",
            "right_after",
            "right_before",
            "other",
        ] = Field(..., description="The question type")
        AnswerType: Literal[
            "binary",
            "multi",
            "open",
        ] = Field(..., description="The answer type")

    prompt = PROMPT_TEMPLATE.format(
        question=row["question"],
        answer=row["answer"],
    )
    chat_response = client.chat.completions.create(
        model="IGNORED",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,  # Leave enough room for thinking and response
        presence_penalty=1.5,
        # Thinking mode (Qwen3):
        temperature=0.6,
        top_p=0.95,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": True},
        },
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "ClassifierResponse",
                "schema": ClassifierResponse.model_json_schema(),
            },
        },
    )
    content = chat_response.choices[0].message.content
    assert content is not None, "The response content should not be None"

    # The result here will be a JSON string
    return content


def decode(response) -> dict[str, Any]:
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        # We print this since it should never happen
        print(f"Failed to decode response: {response}, error: {e}")
        return defaultdict(lambda: pd.NA)


if __name__ == "__main__":
    # Args parser
    parser = ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "file",
        type=str,
        help="Path to the input CSV file containing a column 'question' and 'answer'. The output will be saved to the same file.",
    )
    args = parser.parse_args()

    csv_path = Path(args.file).absolute()
    print(f"Using input path: {csv_path}")
    assert csv_path.exists()

    # Name taken from LAST column of `docker container ls` (NOT the id, but the name)
    # server_url = "http://docker-ts-qa-sg-run-2e4f39436b32:30000"
    server_url = "http://172.18.0.2:30000"
    set_default_backend(RuntimeEndpoint(server_url))  # Checks the connection
    print(f"Successfully connected to server at {server_url}")
    client = OpenAI(api_key="EMPTY", base_url=server_url + "/v1")

    # Handle Data
    df = pd.read_csv(csv_path)

    # Add a results column if not existing and prefill with NA
    if "question_type" not in df.columns:
        df["question_type"] = pd.NA
        df["answer_type"] = pd.NA

    def process_row(row) -> dict[str, str]:
        # We do not want to re-evaluate unnecessarily
        if pd.notna(row["question_type"]):
            return {
                "QuestionType": row["question_type"],
                "AnswerType": row["answer_type"],
            }

        # Else, we need to actually ask the judge
        try:
            string_answer = run_classifier(row, client)

        except openai.InternalServerError as e:  # This sometimes fails
            if "All retry attempts failed" in str(e):
                # Retry once
                sleep(10)
                string_answer = run_classifier(row, client)
            else:
                raise e

        # Parse the JSON
        return decode(string_answer)

    chunk_size = 2_000
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task_chunks = progress.add_task(
            "[green]Processing chunks...", total=len(df) // chunk_size + 1
        )

        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = {}
            for start_idx in range(0, len(df), chunk_size):
                end_idx = min(start_idx + chunk_size, len(df))
                chunk = df.iloc[start_idx:end_idx]

                task_in_chunk = progress.add_task(
                    "[cyan]Processing rows in chunk...", total=chunk_size
                )

                futures = {
                    idx: executor.submit(process_row, row)
                    for idx, row in chunk.iterrows()
                }
                for idx, future in futures.items():
                    parsed = future.result()

                    # Write result back into the dataframe
                    # print(f"Row {idx} classified as {parsed.get('QuestionType', pd.NA)}, {parsed.get('AnswerType', pd.NA)}")
                    df.at[idx, "question_type"] = parsed.get("QuestionType", pd.NA)
                    df.at[idx, "answer_type"] = parsed.get("AnswerType", pd.NA)

                    progress.update(task_in_chunk, advance=1)

                # Save progress after processing each chunk
                df.to_csv(csv_path, index=True)

                progress.remove_task(task_in_chunk)
                progress.update(task_chunks, advance=1)
