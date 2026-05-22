from accident_vlm.training import evaluate_actor_detection_map, train_custom_yolo, write_yolo_dataset_yaml


class FakeYolo:
    def __init__(self, model_name):
        self.model_name = model_name
        self.train_kwargs = None
        self.val_kwargs = None

    def train(self, **kwargs):
        self.train_kwargs = kwargs
        return {"save_dir": "/tmp/runs/train"}

    def val(self, **kwargs):
        self.val_kwargs = kwargs

        class Box:
            map = 0.91
            map50 = 0.96

        class Results:
            box = Box()

        return Results()


def test_write_yolo_dataset_yaml_contains_accident_taxonomy(tmp_path) -> None:
    yaml_path = write_yolo_dataset_yaml(
        dataset_root=tmp_path / "dataset",
        output_path=tmp_path / "dataset.yaml",
    )

    text = yaml_path.read_text(encoding="utf-8")
    assert "kickboard" in text
    assert "traffic_light" in text
    assert "path:" in text


def test_train_custom_yolo_invokes_ultralytics_train(monkeypatch, tmp_path) -> None:
    created = {}

    def fake_yolo(model_name):
        created["model"] = FakeYolo(model_name)
        return created["model"]

    monkeypatch.setattr("accident_vlm.training._load_yolo_class", lambda: fake_yolo)

    result = train_custom_yolo(
        base_model="yolov8x.pt",
        dataset_yaml=tmp_path / "dataset.yaml",
        epochs=5,
        imgsz=960,
        device="0",
        project=tmp_path / "runs",
    )

    assert created["model"].train_kwargs["data"] == str(tmp_path / "dataset.yaml")
    assert created["model"].train_kwargs["epochs"] == 5
    assert result["training_status"] == "submitted"


def test_evaluate_actor_detection_map_extracts_map_metrics(monkeypatch, tmp_path) -> None:
    created = {}

    def fake_yolo(model_name):
        created["model"] = FakeYolo(model_name)
        return created["model"]

    monkeypatch.setattr("accident_vlm.training._load_yolo_class", lambda: fake_yolo)

    metrics = evaluate_actor_detection_map(
        model_path=tmp_path / "best.pt",
        dataset_yaml=tmp_path / "dataset.yaml",
        split="test",
    )

    assert metrics["actor_detection_mAP"] == 0.91
    assert metrics["actor_detection_mAP50"] == 0.96
    assert created["model"].val_kwargs["split"] == "test"
