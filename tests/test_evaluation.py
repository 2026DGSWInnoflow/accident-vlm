import json

from accident_vlm.evaluation import evaluate_result_against_label, load_dataset_labels, summarize_evaluation


def test_load_dataset_labels_indexes_by_video_name(tmp_path) -> None:
    label_path = tmp_path / "label.json"
    label_path.write_text(
        json.dumps({"video": {"video_name": "sample_001", "accident_object": 1}}),
        encoding="utf-8",
    )

    labels = load_dataset_labels(tmp_path)

    assert labels["sample_001"]["video"]["accident_object"] == 1


def test_evaluate_result_against_label_scores_object_match() -> None:
    result = {
        "objective_summary": "자차와 보행자가 확인됨",
        "timeline": [{"event": "보행자 진입"}],
        "traffic_control": {"signal": {"value": "적색"}},
        "uncertainties": ["야간"],
        "rag_hints": {"accident_type": "차대보행자"},
    }
    label = {"video": {"accident_object": 1}}

    evaluation = evaluate_result_against_label(result, label)

    assert evaluation["checks"]["accident_object_match"] is True
    assert evaluation["quality_bucket"] == "high"


def test_summarize_evaluation_estimates_usable_per_100() -> None:
    summary = summarize_evaluation(
        [
            {"quality_bucket": "high"},
            {"quality_bucket": "medium"},
            {"quality_bucket": "low"},
            {"quality_bucket": "low"},
        ]
    )

    assert summary["estimated_usable_count_per_100"] == 50
    assert summary["estimated_high_quality_count_per_100"] == 25
