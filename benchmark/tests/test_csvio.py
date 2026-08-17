from quants_ablation.csvio import read_prediction_csv, write_prediction_csv


def test_unicode_csv_round_trip(tmp_path):
    rows = [
        {
            "sample_id": 27001,
            "question_id": 4,
            "prediction_text": 'line 1\nline 2, "quoted" 🧪',
        },
        {"sample_id": 27000, "question_id": 0, "prediction_text": "café 東京"},
    ]
    destination = tmp_path / "ts_only_open_test.csv"
    write_prediction_csv(destination, rows, "test")
    raw = destination.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert raw.startswith(b"sample_id,question_id,prediction_text\r\n")
    assert read_prediction_csv(destination) == [
        {"sample_id": 27000, "question_id": 0, "prediction_text": "café 東京"},
        {
            "sample_id": 27001,
            "question_id": 4,
            "prediction_text": 'line 1\nline 2, "quoted" 🧪',
        },
    ]
