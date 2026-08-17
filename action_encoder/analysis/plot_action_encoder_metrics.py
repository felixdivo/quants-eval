#!/usr/bin/env python3
"""Plot post-hoc position and per-class metrics from action_metrics.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--view",
        choices=("unique_sample", "qa_row_weighted"),
        default="unique_sample",
    )
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    payload = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    view = payload["views"][args.view]
    positions = view["by_position"]
    classes = [row for row in view["per_class"] if row["support"]]

    fig, (position_axis, class_axis) = plt.subplots(
        1,
        2,
        figsize=(15, 5.4),
        gridspec_kw={"width_ratios": [0.8, 2.2]},
        constrained_layout=True,
    )
    position_axis.plot(
        [row["segment_id"] + 1 for row in positions],
        [100 * row["accuracy"] for row in positions],
        marker="o",
        linewidth=2.5,
        color="#3F96C5",
    )
    position_axis.set_title("Segment accuracy by position")
    position_axis.set_xlabel("Segment position")
    position_axis.set_ylabel("Accuracy (%)")
    position_axis.set_xticks([1, 2, 3, 4])
    position_axis.grid(axis="y", alpha=0.25)

    class_axis.barh(
        [row["action_name"] for row in classes],
        [100 * row["f1"] for row in classes],
        color="#41B696",
    )
    class_axis.set_title("Per-action F1")
    class_axis.set_xlabel("F1 (%)")
    class_axis.grid(axis="x", alpha=0.25)
    class_axis.invert_yaxis()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    if args.output.suffix.lower() != ".pdf":
        fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
