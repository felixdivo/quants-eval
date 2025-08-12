"""
```shell
docker compose -f .docker/compose.yml -f .docker/compose.dev.yml -f .docker/compose.cuda.yml run --build ts-qa-sg

-> in container
export CUDA_VISIBLE_DEVICES=2,3
python -m sglang.launch_server --model-path Qwen/Qwen3-8B-AWQ --port 30000 --host 0.0.0.0 --dp-size 2 --reasoning-parser qwen3
```
"""

from time import sleep
from typing import Any, Literal
from pathlib import Path
import json
from collections import defaultdict
import shutil
import warnings

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


def run_judge(row: pd.Series, client: OpenAI) -> str:
    """Run the LLM judge on a single row of the evaluation dataset.

    Args:
        row: A row from the evaluation dataset containing the scene description, question, reference answer,
             and system answer.
        client: An instance of the OpenAI client to interact with the LLM judge.

    Returns:
        str: The JSON response from the LLM judge containing the rating and rationale.
    """

    JUDGE_PROMPT_TEMPLATE = """You will be given a scene_description with timesamps and a scene_question regarding that scene. You will also be given a reference_answer and system_answer couple.
    Your task is to provide a TotalRating scoring how well the system_answer answers the user concerns expressed in the scene_question. The reference_answer is provided as reference for a very good answer.
    Give your answer on a scale of 1 to 3, where 1 means that the system_answer is not helpful at all, and 3 means that the system_answer completely and helpfully addresses the scene_question.
    You will also provide a brief rationale for your rating.

    Here is the scale you should use to build your answer:
    1: The system_answer is terrible: the answer is factually incorrect or corrupt in some way (e.g., empty or nonsensical)
    2: The system_answer is bad: somewhat correctly answers the scene_question like the reference_answer, yet contains additional useless/ambiguous/wrong information or is incomplete
    3: The system_answer is good: correctly answers the scene_question and is helpful in the context of the scene

    Provide your assessment in the following JSON format:
    {{
        "BriefRationale": "<your rationale, at most 3 sentences>",
        "TotalRating": <your rating>
    }}

    You MUST provide values for "BriefRationale" and "TotalRating" in your answer.
    Do NOT respond with any other text in your answer.
    Immediately, start with the JSON object, without any additional text or explanation.
    Make sure to output complete and valid JSON.

    Now here are the question and answer:
    - scene_description: {scene_description}
    - scene_question: {scene_question}
    - reference_answer: {reference_answer}
    - system_answer: {system_answer}

    Provide your assessment.
    """

    max_rationale = 500

    class JudgeResponse(BaseModel):
        BriefRationale: str = Field(
            ...,
            max_length=max_rationale,
            description="Your rationale, at most 3 sentences",
        )
        TotalRating: Literal[1, 2, 3] = Field(
            ...,
            description="Your rating on a scale of 1 to 3",
        )

    scene_description = row["textual_description"]
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        scene_description=scene_description,
        scene_question=row["question"],
        reference_answer=row["answer_text"],
        system_answer=row["prediction_text"],
    )
    chat_response = client.chat.completions.create(
        model="IGNORED",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8192,  # Leave enough room for thinking and response
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
                "name": "JudgeResponse",
                "schema": JudgeResponse.model_json_schema(),
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
    base_path = Path(__file__).absolute().parent / "data"
    print(f"Using base path: {base_path}")
    assert base_path.is_dir() and base_path.exists()

    # Name taken from LAST column of `docker container ls` (NOT the id, but the name)
    # server_url = "http://docker-ts-qa-sg-run-2e4f39436b32:30000"
    server_url = "http://172.19.0.3:30000"
    set_default_backend(RuntimeEndpoint(server_url))  # Checks the connection
    print(f"Successfully connected to server at {server_url}")
    client = OpenAI(api_key="EMPTY", base_url=server_url + "/v1")

    # Handle Data
    file_source = base_path / "all_joined.h5"
    file_target = base_path / "all_joined_judged.h5"
    if not file_target.exists():
        shutil.copyfile(file_source, file_target)

    df = pd.read_hdf(file_target, key="data")

    # Add a results column if not existing and prefill with NA
    if "llm_judge_raw" not in df.columns:
        df["llm_judge_raw"] = pd.NA
        df["llm_judge_rationale"] = pd.NA
        df["llm_judge_score"] = pd.NA

    def process_row(row) -> tuple[str, dict[str, Any]]:
        # We only do open, and we do not want to re-evaluate unnecessaryly
        if row["answer_type"] != "open" or pd.notna(row["llm_judge_raw"]):
            return row["llm_judge_raw"], {
                "BriefRationale": row["llm_judge_rationale"],
                "TotalRating": row["llm_judge_score"],
            }

        # Else, we need to actually ask the judge
        try:
            string_answer = run_judge(row, client)

        except openai.InternalServerError as e:  # This sometimes fails
            if "All retry attempts failed" in str(e):
                # Retry once
                sleep(10)
                string_answer = run_judge(row, client)
            else:
                raise e

        # Parse the JSON
        return string_answer, decode(string_answer)

    chunk_size = 20_000
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
                    raw, parsed = future.result()

                    # Write result back into the dataframe
                    df.at[idx, "llm_judge_raw"] = raw
                    df.at[idx, "llm_judge_rationale"] = parsed["BriefRationale"]
                    df.at[idx, "llm_judge_score"] = parsed["TotalRating"]

                    progress.update(task_in_chunk, advance=1)

                # Save progress after processing each chunk
                with warnings.catch_warnings():
                    warnings.simplefilter(
                        "ignore", category=pd.errors.PerformanceWarning
                    )
                    df.to_hdf(file_target, key="data", mode="w", complevel=7)

                progress.remove_task(task_in_chunk)
                progress.update(task_chunks, advance=1)
