import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quants_ablation.data import load_ts_prompt_cache


def _write_cache(path, sample_ids):
    pq.write_table(
        pa.table(
            {
                "sample_id": sample_ids,
                "input_ids": [[1, 2] for _ in sample_ids],
                "trajectory_sha256": ["a" * 64 for _ in sample_ids],
                "token_count": [2 for _ in sample_ids],
            }
        ),
        path,
    )


def test_ts_prompt_cache_preserves_prevalidated_order_without_copy(tmp_path):
    path = tmp_path / "cache.parquet"
    _write_cache(path, [10, 11])
    dataset = load_ts_prompt_cache(path)
    assert dataset[0]["sample_id"] == 10
    assert dataset[1]["sample_id"] == 11


def test_ts_prompt_cache_rejects_unsorted_ids(tmp_path):
    path = tmp_path / "cache.parquet"
    _write_cache(path, [11, 10])
    with pytest.raises(ValueError, match="not strictly sorted"):
        load_ts_prompt_cache(path)
