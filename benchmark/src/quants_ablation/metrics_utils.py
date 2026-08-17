from __future__ import annotations

import re


YES_NO_PATTERN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
MULTI_PATTERN = re.compile(r"(?<![A-Za-z0-9])([ABC])(?![A-Za-z0-9])")


def parse_binary(text: str) -> int | None:
    matches = {match.lower() for match in YES_NO_PATTERN.findall(text)}
    if matches == {"yes"}:
        return 1
    if matches == {"no"}:
        return 0
    return None


def parse_multi(text: str) -> int | None:
    matches = set(MULTI_PATTERN.findall(text))
    if len(matches) != 1:
        return None
    return {"A": 0, "B": 1, "C": 2}[next(iter(matches))]
