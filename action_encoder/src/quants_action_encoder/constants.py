"""Pinned inputs and xLSTMMixer action-encoder constants."""

from __future__ import annotations

DATASET_ID = "dasyd/quants"
DATASET_REVISION = "0e78849313d3b043a0b08697dc004fb6cba15df9"

# xLSTMMixer is not part of an upstream Transformers release, so both the fork
# and its exact source revision are pinned.
TRANSFORMERS_REPOSITORY = "https://github.com/mauricekraus/transformers.git"
TRANSFORMERS_REVISION = "e97fc33f97155b79658af37fde29b39b5139e066"

TASKS = ("binary", "multi", "open")
SPLITS = ("train", "val", "test")
SPLIT_ID_RANGES = {
    "train": (0, 23_999),
    "val": (24_000, 26_999),
    "test": (27_000, 29_999),
}
EXPECTED_ROWS = {
    "binary": {"train": 49_454, "val": 6_235, "test": 6_115},
    "multi": {"train": 34_927, "val": 4_326, "test": 4_346},
    "open": {"train": 35_619, "val": 4_439, "test": 4_539},
}
EXPECTED_PARQUET_SHARDS = {
    "binary": {"train": 13, "val": 2, "test": 2},
    "multi": {"train": 9, "val": 2, "test": 2},
    "open": {"train": 9, "val": 2, "test": 2},
}

TRAJECTORY_SHAPE = (320, 24, 3)
SEGMENT_COUNT = 4
SEGMENT_LENGTH = 80
INPUT_CHANNELS = 72
NUM_ACTIONS = 19

# Ordering used by both training labels and the action-name decoder, and verified
# against every label exposed by the released dataset. IDs are positional and
# must not be alphabetically reordered.
ACTION_NAMES = (
    "holding a baby",
    "shaking hands",
    "running",
    "jumping once",
    "punching",
    "golfing (swinging a club)",
    "drinking with the left hand",
    "skipping rope",
    "dancing",
    "waving",
    "playing guitar",
    "bowing",
    "kicking a ball",
    "throwing a ball",
    "T-posing",
    "catching a ball",
    "picking something up with both hands",
    "sitting down",
    "eating with the right hand",
)
ACTION_TO_ID = {name: action_id for action_id, name in enumerate(ACTION_NAMES)}

if len(ACTION_NAMES) != NUM_ACTIONS or len(ACTION_TO_ID) != NUM_ACTIONS:
    raise RuntimeError("action labels must contain exactly 19 unique names")
