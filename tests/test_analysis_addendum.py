from __future__ import annotations

import json
from pathlib import Path


def test_scored_json_has_expected_shape_and_60_items():
    root = Path(__file__).resolve().parents[1]
    scored = root / "compare" / "llm_compare_20260401_131033_scored.json"
    data = json.loads(scored.read_text(encoding="utf-8"))

    assert data["meta"]["rows"] == 60
    assert len(data["items"]) == 60

    first = data["items"][0]
    assert "question" in first and "ground_truth" in first
    for m in ("llama2", "mistral"):
        assert "answer" in first[m]
        assert "scores" in first[m]
        for dim in data["meta"]["dimensions"]:
            assert 0 <= first[m]["scores"][dim] <= 5


def test_addendum_generator_imports_and_classifies():
    # lightweight import test + basic function checks
    from compare.generate_analysis_addendum import classify_question, extract_error_tags

    it = {
        "question": "When did X happen?",
        "ground_truth": "20260101_10:00",
        "llama2": {"answer": "20260101_11:00", "scores": {"correctness": 3, "hallucination": 4}, "debug": {"gold_type": "datetime", "pred_datetime": "20260101_11:00", "gold_datetime": "20260101_10:00"}},
        "mistral": {"answer": "I don't know", "scores": {"correctness": 0, "hallucination": 5}, "debug": {"gold_type": "datetime", "pred_datetime": None, "gold_datetime": "20260101_10:00"}},
        "better_model": "tie",
    }

    assert classify_question(it["question"], it["ground_truth"]) == "time"
    tags = extract_error_tags(it, "llama2")
    assert "time_offset_same_date" in tags
    tags2 = extract_error_tags(it, "mistral")
    assert "idk" in tags2

