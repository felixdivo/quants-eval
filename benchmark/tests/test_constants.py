import json
import os
from pathlib import Path
import subprocess
import sys


_SCRIPT = """
import json
from quants_ablation.constants import (
    ARTIFACTS_ROOT,
    CACHE_ROOT,
    DATA_ROOT,
    MODEL_ROOT,
    PROJECT_ROOT,
    RESULTS_ROOT,
)
print(json.dumps({
    "project": str(PROJECT_ROOT),
    "data": str(DATA_ROOT),
    "cache": str(CACHE_ROOT),
    "results": str(RESULTS_ROOT),
    "artifacts": str(ARTIFACTS_ROOT),
    "models": str(MODEL_ROOT),
}))
"""


def _read_roots(root_override: str | None) -> dict[str, str]:
    environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[1] / "src"
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(source_root), *([current_pythonpath] if current_pythonpath else [])]
    )
    if root_override is None:
        environment.pop("QUANTS_ABLATION_ROOT", None)
    else:
        environment["QUANTS_ABLATION_ROOT"] = root_override
    completed = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return json.loads(completed.stdout)


def test_default_project_root_is_benchmark_directory():
    expected = Path(__file__).resolve().parents[1]
    roots = _read_roots(None)
    assert Path(roots["project"]) == expected


def test_environment_override_relocates_all_runtime_roots(tmp_path):
    override = tmp_path / "nested" / ".." / "relocated"
    expected = override.resolve()
    roots = _read_roots(str(override))
    assert Path(roots["project"]) == expected
    assert Path(roots["data"]) == expected / "data" / "raw" / "quants"
    assert Path(roots["cache"]) == expected / "data" / "prepared"
    assert Path(roots["results"]) == expected / "results"
    assert Path(roots["artifacts"]) == expected / "artifacts"
    assert Path(roots["models"]) == expected / "models"
