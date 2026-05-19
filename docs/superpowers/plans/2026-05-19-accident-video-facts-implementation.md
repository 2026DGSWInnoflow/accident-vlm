# Accident Video Fact Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python pipeline that converts dashcam accident videos into objective, evidence-linked JSON facts for downstream legal/RAG retrieval.

**Architecture:** The implementation is contract-first: Pydantic schemas define stable module inputs and outputs, then each preprocessing module writes structured observations into an evidence package, and Qwen VLM composes the final fact JSON under schema and hallucination guards. Heavy CV/OCR models are wrapped behind interfaces so accuracy-focused models can be upgraded without changing the RAG contract.

**Tech Stack:** Python 3.11+, Pydantic v2, OpenCV, ffmpeg/ffprobe, NumPy, pytest, optional PaddleOCR/EasyOCR, optional YOLO/RT-DETR, optional ByteTrack/BoT-SORT, optional Qwen multimodal backend.

---

## File Structure

Create this structure:

```text
pyproject.toml
README.md
src/accident_vlm/
  __init__.py
  cli.py
  config.py
  pipeline.py
  schemas/
    __init__.py
    common.py
    final_output.py
    preprocessing.py
  modules/
    __init__.py
    ingestion.py
    frame_selection.py
    ocr.py
    scene.py
    actor_tracking.py
    road_geometry.py
    speed_distance.py
    traffic_control.py
    event_detection.py
    evidence_builder.py
    vlm_composer.py
    schema_guard.py
  utils/
    __init__.py
    timecode.py
    files.py
tests/
  fixtures/
    sample_precomputed.json
  test_schemas.py
  test_timecode.py
  test_ingestion.py
  test_frame_selection.py
  test_speed_distance.py
  test_schema_guard.py
  test_pipeline_contract.py
```

Responsibility map:

- `schemas/`: source of truth for JSON contracts and enums.
- `modules/`: one focused preprocessing or composition module per pipeline stage.
- `pipeline.py`: orchestration only; it should not contain CV logic.
- `cli.py`: command-line entrypoint.
- `utils/`: shared pure helpers.
- `tests/`: contract tests and deterministic unit tests.

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/accident_vlm/__init__.py`
- Create: `src/accident_vlm/config.py`
- Create: `tests/test_timecode.py`
- Create: `src/accident_vlm/utils/__init__.py`
- Create: `src/accident_vlm/utils/timecode.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "accident-vlm"
version = "0.1.0"
description = "Dashcam accident video fact extraction pipeline for RAG"
requires-python = ">=3.11"
dependencies = [
  "pydantic>=2.7",
  "opencv-python>=4.9",
  "numpy>=1.26",
  "typer>=0.12",
  "rich>=13.7"
]

[project.optional-dependencies]
ocr = ["paddleocr>=2.7", "easyocr>=1.7"]
dev = ["pytest>=8.0", "pytest-cov>=5.0", "ruff>=0.5"]

[project.scripts]
accident-vlm = "accident_vlm.cli:app"

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"
```

- [ ] **Step 2: Create `README.md`**

```markdown
# Accident VLM

Dashcam accident video fact extraction pipeline.

The module produces objective, evidence-linked JSON for downstream RAG retrieval.
It does not determine legal liability, fault ratio, traffic-law violation, offender, or victim.

Primary output schema: `accident_video_facts.v1`.
```

- [ ] **Step 3: Create package marker files**

```python
# src/accident_vlm/__init__.py
__all__ = ["__version__"]

__version__ = "0.1.0"
```

```python
# src/accident_vlm/utils/__init__.py
```

- [ ] **Step 4: Write failing tests for timecode conversion**

```python
# tests/test_timecode.py
from accident_vlm.utils.timecode import frame_to_timecode, seconds_to_timecode


def test_seconds_to_timecode():
    assert seconds_to_timecode(65.432) == "01:05.432"


def test_frame_to_timecode():
    assert frame_to_timecode(frame_index=90, fps=30) == "00:03.000"
```

- [ ] **Step 5: Run tests and verify failure**

Run: `pytest tests/test_timecode.py -v`

Expected: fails with `ModuleNotFoundError` or missing function.

- [ ] **Step 6: Implement timecode helpers**

```python
# src/accident_vlm/utils/timecode.py
def seconds_to_timecode(seconds: float) -> str:
    if seconds < 0:
        raise ValueError("seconds must be non-negative")
    minutes = int(seconds // 60)
    remaining = seconds - (minutes * 60)
    whole_seconds = int(remaining)
    millis = int(round((remaining - whole_seconds) * 1000))
    if millis == 1000:
        whole_seconds += 1
        millis = 0
    return f"{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def frame_to_timecode(frame_index: int, fps: float) -> str:
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    if fps <= 0:
        raise ValueError("fps must be positive")
    return seconds_to_timecode(frame_index / fps)
```

- [ ] **Step 7: Add config model**

```python
# src/accident_vlm/config.py
from pathlib import Path

from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    output_dir: Path = Field(default=Path("outputs"))
    regular_frame_interval_sec: float = Field(default=1.0, gt=0)
    pre_event_window_sec: float = Field(default=5.0, gt=0)
    post_event_window_sec: float = Field(default=3.0, gt=0)
    enable_ocr: bool = True
    enable_vlm: bool = False
```

- [ ] **Step 8: Run tests and verify pass**

Run: `pytest tests/test_timecode.py -v`

Expected: 2 passed.

## Task 2: Schema Contracts and Enums

**Files:**
- Create: `src/accident_vlm/schemas/__init__.py`
- Create: `src/accident_vlm/schemas/common.py`
- Create: `src/accident_vlm/schemas/preprocessing.py`
- Create: `src/accident_vlm/schemas/final_output.py`
- Create: `tests/test_schemas.py`

- [ ] **Step 1: Write schema tests**

```python
# tests/test_schemas.py
import pytest
from pydantic import ValidationError

from accident_vlm.schemas.common import Confidence, EvidenceField, Status
from accident_vlm.schemas.final_output import AccidentFactOutput, AccidentType, SceneType


def test_evidence_field_accepts_unknown_without_evidence():
    field = EvidenceField(
        value="확인불가",
        status=Status.UNKNOWN,
        confidence=Confidence.UNKNOWN,
        source=[],
        evidence=[],
        note="영상에서 확인되지 않음",
    )
    assert field.value == "확인불가"


def test_evidence_field_rejects_observed_without_source():
    with pytest.raises(ValidationError):
        EvidenceField(value="주간", status=Status.OBSERVED, confidence=Confidence.HIGH)


def test_final_output_minimal_contract():
    output = AccidentFactOutput(
        scene_type=EvidenceField(
            value=SceneType.ROAD,
            status=Status.OBSERVED,
            confidence=Confidence.MEDIUM,
            source=["visual"],
            evidence=["frame_000030"],
        ),
        rag_hints={"accident_type": AccidentType.VEHICLE_TO_VEHICLE, "scenario_keywords": []},
        objective_summary="자차와 상대 차량의 접촉이 관찰됨.",
    )
    assert output.schema_version == "accident_video_facts.v1"
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_schemas.py -v`

Expected: fails because schema modules do not exist.

- [ ] **Step 3: Implement common schema primitives**

```python
# src/accident_vlm/schemas/common.py
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class Status(StrEnum):
    OBSERVED = "observed"
    COMPUTED = "computed"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class EvidenceField(BaseModel):
    value: Any
    raw: str | None = None
    status: Status
    confidence: Confidence
    source: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    note: str | None = None

    @model_validator(mode="after")
    def require_source_for_non_unknown(self) -> "EvidenceField":
        if self.status != Status.UNKNOWN and not self.source:
            raise ValueError("source is required when status is not unknown")
        return self
```

- [ ] **Step 4: Implement preprocessing schemas**

```python
# src/accident_vlm/schemas/preprocessing.py
from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    duration_sec: float
    fps: float
    resolution: str
    frame_count: int
    has_audio: bool


class InputQuality(BaseModel):
    blur: str
    brightness: str
    night_noise: str
    camera_shake: str
    occlusion: str
    analysis_reliability: str


class SelectedFrame(BaseModel):
    id: str
    time: str
    frame_index: int
    path: str | None = None
    purpose: str


class PipelineContext(BaseModel):
    video_path: str
    video_metadata: VideoMetadata | None = None
    input_quality: InputQuality | None = None
    selected_frames: list[SelectedFrame] = Field(default_factory=list)
    ocr_observations: list[dict] = Field(default_factory=list)
    tracks: list[dict] = Field(default_factory=list)
    road_geometry: dict = Field(default_factory=dict)
    speed_and_distance: dict = Field(default_factory=dict)
    traffic_control: dict = Field(default_factory=dict)
    event_candidates: list[dict] = Field(default_factory=list)
    evidence_package: dict = Field(default_factory=dict)
```

- [ ] **Step 5: Implement final output schemas**

```python
# src/accident_vlm/schemas/final_output.py
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from accident_vlm.schemas.common import EvidenceField


class SceneType(StrEnum):
    INTERSECTION = "교차로"
    CROSSWALK = "횡단보도"
    ROAD = "일반도로"
    HIGHWAY = "고속도로"
    PARKING_LOT = "주차장"
    ALLEY = "골목길"
    UNKNOWN = "확인불가"


class AccidentType(StrEnum):
    VEHICLE_TO_VEHICLE = "차대차"
    VEHICLE_TO_PEDESTRIAN = "차대보행자"
    VEHICLE_TO_BICYCLE = "차대자전거"
    VEHICLE_TO_MOTORCYCLE = "차대이륜차"
    VEHICLE_TO_FACILITY = "차대시설물"
    SINGLE_VEHICLE = "단독사고"
    NON_CONTACT = "비접촉사고"
    MULTI_COLLISION = "다중추돌"
    UNKNOWN = "확인불가"


class AccidentFactOutput(BaseModel):
    schema_version: str = "accident_video_facts.v1"
    input_quality: dict[str, Any] = Field(default_factory=dict)
    scene_type: EvidenceField
    road_conditions: dict[str, Any] = Field(default_factory=dict)
    traffic_control: dict[str, Any] = Field(default_factory=dict)
    actors: list[dict[str, Any]] = Field(default_factory=list)
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    collision: dict[str, Any] = Field(default_factory=dict)
    speed_and_distance: dict[str, Any] = Field(default_factory=dict)
    uncertainties: list[str] = Field(default_factory=list)
    evidence_index: dict[str, Any] = Field(default_factory=dict)
    rag_hints: dict[str, Any]
    objective_summary: str
```

- [ ] **Step 6: Export schemas**

```python
# src/accident_vlm/schemas/__init__.py
from accident_vlm.schemas.common import Confidence, EvidenceField, Status
from accident_vlm.schemas.final_output import AccidentFactOutput, AccidentType, SceneType
from accident_vlm.schemas.preprocessing import PipelineContext, SelectedFrame, VideoMetadata

__all__ = [
    "AccidentFactOutput",
    "AccidentType",
    "Confidence",
    "EvidenceField",
    "PipelineContext",
    "SceneType",
    "SelectedFrame",
    "Status",
    "VideoMetadata",
]
```

- [ ] **Step 7: Run tests and verify pass**

Run: `pytest tests/test_schemas.py -v`

Expected: 3 passed.

## Task 3: Video Ingestion Module

**Files:**
- Create: `src/accident_vlm/modules/__init__.py`
- Create: `src/accident_vlm/modules/ingestion.py`
- Create: `tests/test_ingestion.py`

- [ ] **Step 1: Write tests using mocked probe output**

```python
# tests/test_ingestion.py
from accident_vlm.modules.ingestion import parse_ffprobe_streams


def test_parse_ffprobe_streams_video_and_audio():
    data = {
        "streams": [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30/1",
                "nb_frames": "300",
                "duration": "10.0",
            },
            {"codec_type": "audio"},
        ]
    }

    metadata = parse_ffprobe_streams(data)

    assert metadata.duration_sec == 10.0
    assert metadata.fps == 30
    assert metadata.resolution == "1920x1080"
    assert metadata.frame_count == 300
    assert metadata.has_audio is True
```

- [ ] **Step 2: Run test and verify failure**

Run: `pytest tests/test_ingestion.py -v`

Expected: fails because `parse_ffprobe_streams` does not exist.

- [ ] **Step 3: Implement ingestion parser**

```python
# src/accident_vlm/modules/__init__.py
```

```python
# src/accident_vlm/modules/ingestion.py
from fractions import Fraction
from typing import Any

from accident_vlm.schemas.preprocessing import VideoMetadata


def parse_ffprobe_streams(data: dict[str, Any]) -> VideoMetadata:
    streams = data.get("streams", [])
    video_stream = next(stream for stream in streams if stream.get("codec_type") == "video")
    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
    fps = float(Fraction(video_stream.get("avg_frame_rate", "0/1")))
    duration = float(video_stream.get("duration", 0.0))
    frame_count_raw = video_stream.get("nb_frames")
    frame_count = int(frame_count_raw) if frame_count_raw else int(round(duration * fps))
    return VideoMetadata(
        duration_sec=duration,
        fps=fps,
        resolution=f"{video_stream['width']}x{video_stream['height']}",
        frame_count=frame_count,
        has_audio=has_audio,
    )
```

- [ ] **Step 4: Run test and verify pass**

Run: `pytest tests/test_ingestion.py -v`

Expected: 1 passed.

- [ ] **Step 5: Add real `ffprobe` wrapper after parser is tested**

Append to `src/accident_vlm/modules/ingestion.py`:

```python
import json
import subprocess
from pathlib import Path


def probe_video(video_path: Path) -> VideoMetadata:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_ffprobe_streams(json.loads(result.stdout))
```

## Task 4: Frame Selection Module

**Files:**
- Create: `src/accident_vlm/modules/frame_selection.py`
- Create: `tests/test_frame_selection.py`

- [ ] **Step 1: Write deterministic frame selection test**

```python
# tests/test_frame_selection.py
from accident_vlm.modules.frame_selection import select_regular_frames


def test_select_regular_frames_every_second():
    frames = select_regular_frames(duration_sec=3.2, fps=30, interval_sec=1.0)

    assert [frame.frame_index for frame in frames] == [0, 30, 60, 90]
    assert [frame.time for frame in frames] == ["00:00.000", "00:01.000", "00:02.000", "00:03.000"]
```

- [ ] **Step 2: Run test and verify failure**

Run: `pytest tests/test_frame_selection.py -v`

Expected: fails because module does not exist.

- [ ] **Step 3: Implement regular frame selection**

```python
# src/accident_vlm/modules/frame_selection.py
from accident_vlm.schemas.preprocessing import SelectedFrame
from accident_vlm.utils.timecode import frame_to_timecode


def select_regular_frames(duration_sec: float, fps: float, interval_sec: float) -> list[SelectedFrame]:
    if duration_sec < 0:
        raise ValueError("duration_sec must be non-negative")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if interval_sec <= 0:
        raise ValueError("interval_sec must be positive")

    frame_indices: list[int] = []
    current = 0.0
    while current <= duration_sec:
        frame_indices.append(int(round(current * fps)))
        current += interval_sec

    return [
        SelectedFrame(
            id=f"frame_{frame_index:06d}",
            time=frame_to_timecode(frame_index, fps),
            frame_index=frame_index,
            purpose="regular_context",
        )
        for frame_index in frame_indices
    ]
```

- [ ] **Step 4: Run test and verify pass**

Run: `pytest tests/test_frame_selection.py -v`

Expected: 1 passed.

## Task 5: Speed and Distance Contract Module

**Files:**
- Create: `src/accident_vlm/modules/speed_distance.py`
- Create: `tests/test_speed_distance.py`

- [ ] **Step 1: Write tests for speed priority policy**

```python
# tests/test_speed_distance.py
from accident_vlm.modules.speed_distance import choose_speed_estimate


def test_choose_ocr_speed_before_geometry_speed():
    estimates = [
        {"actor_id": "ego_vehicle", "numeric_kmh": 42, "method": "bev_tracking_estimate", "confidence": "medium"},
        {"actor_id": "ego_vehicle", "numeric_kmh": 47, "method": "ocr_overlay", "confidence": "high"},
    ]

    selected = choose_speed_estimate(estimates, actor_id="ego_vehicle")

    assert selected["numeric_kmh"] == 47
    assert selected["method"] == "ocr_overlay"


def test_choose_unknown_when_no_supported_speed():
    selected = choose_speed_estimate([], actor_id="ego_vehicle")

    assert selected["value"] == "모름"
    assert selected["method"] == "not_available"
    assert selected["confidence"] == "unknown"
```

- [ ] **Step 2: Run test and verify failure**

Run: `pytest tests/test_speed_distance.py -v`

Expected: fails because function does not exist.

- [ ] **Step 3: Implement speed priority selection**

```python
# src/accident_vlm/modules/speed_distance.py
SPEED_METHOD_PRIORITY = {
    "metadata": 0,
    "gps": 0,
    "obd": 0,
    "ocr_overlay": 1,
    "bev_tracking_estimate": 2,
    "relative_motion_only": 3,
}


def choose_speed_estimate(estimates: list[dict], actor_id: str) -> dict:
    actor_estimates = [estimate for estimate in estimates if estimate.get("actor_id") == actor_id]
    if not actor_estimates:
        return {
            "actor_id": actor_id,
            "value": "모름",
            "numeric_kmh": None,
            "range_kmh": None,
            "method": "not_available",
            "confidence": "unknown",
        }
    return sorted(
        actor_estimates,
        key=lambda estimate: SPEED_METHOD_PRIORITY.get(estimate.get("method", ""), 99),
    )[0]
```

- [ ] **Step 4: Run test and verify pass**

Run: `pytest tests/test_speed_distance.py -v`

Expected: 2 passed.

## Task 6: Schema Guard and Judgment Filter

**Files:**
- Create: `src/accident_vlm/modules/schema_guard.py`
- Create: `tests/test_schema_guard.py`

- [ ] **Step 1: Write tests for forbidden legal terms**

```python
# tests/test_schema_guard.py
from accident_vlm.modules.schema_guard import find_forbidden_terms, sanitize_summary


def test_find_forbidden_terms():
    text = "상대 차량의 신호위반과 과실이 있는 것으로 보임"

    assert find_forbidden_terms(text) == ["과실", "신호위반"]


def test_sanitize_summary_replaces_forbidden_terms():
    text = "상대 차량의 과실이 관찰됨"

    sanitized = sanitize_summary(text)

    assert "과실" not in sanitized
    assert sanitized == "상대 차량의 [법적 판단 표현 제거]이 관찰됨"
```

- [ ] **Step 2: Run test and verify failure**

Run: `pytest tests/test_schema_guard.py -v`

Expected: fails because module does not exist.

- [ ] **Step 3: Implement forbidden term filter**

```python
# src/accident_vlm/modules/schema_guard.py
FORBIDDEN_LEGAL_TERMS = [
    "가해",
    "피해",
    "과실",
    "위반",
    "불법",
    "책임",
    "주의의무",
    "신호위반",
    "안전거리 미확보",
]


def find_forbidden_terms(text: str) -> list[str]:
    return [term for term in FORBIDDEN_LEGAL_TERMS if term in text]


def sanitize_summary(text: str) -> str:
    sanitized = text
    for term in FORBIDDEN_LEGAL_TERMS:
        sanitized = sanitized.replace(term, "[법적 판단 표현 제거]")
    return sanitized
```

- [ ] **Step 4: Run test and verify pass**

Run: `pytest tests/test_schema_guard.py -v`

Expected: 2 passed.

- [ ] **Step 5: Add output validation wrapper**

Append to `src/accident_vlm/modules/schema_guard.py`:

```python
from accident_vlm.schemas.final_output import AccidentFactOutput


def validate_final_output(payload: dict) -> AccidentFactOutput:
    output = AccidentFactOutput.model_validate(payload)
    forbidden = find_forbidden_terms(output.objective_summary)
    if forbidden:
        output.objective_summary = sanitize_summary(output.objective_summary)
        output.uncertainties.append(
            f"법적 판단 표현이 제거됨: {', '.join(forbidden)}"
        )
    return output
```

## Task 7: Evidence Builder and Pipeline Contract

**Files:**
- Create: `src/accident_vlm/modules/evidence_builder.py`
- Create: `src/accident_vlm/pipeline.py`
- Create: `tests/test_pipeline_contract.py`

- [ ] **Step 1: Write pipeline contract test**

```python
# tests/test_pipeline_contract.py
from accident_vlm.pipeline import build_pre_vlm_context
from accident_vlm.schemas.preprocessing import VideoMetadata


def test_build_pre_vlm_context_contains_regular_frames():
    context = build_pre_vlm_context(
        video_path="sample.mp4",
        metadata=VideoMetadata(
            duration_sec=2.0,
            fps=30,
            resolution="1920x1080",
            frame_count=60,
            has_audio=False,
        ),
    )

    assert context.video_path == "sample.mp4"
    assert context.video_metadata is not None
    assert len(context.selected_frames) == 3
    assert context.evidence_package["precomputed_facts"]["metadata"]["fps"] == 30
```

- [ ] **Step 2: Run test and verify failure**

Run: `pytest tests/test_pipeline_contract.py -v`

Expected: fails because pipeline is missing.

- [ ] **Step 3: Implement evidence package builder**

```python
# src/accident_vlm/modules/evidence_builder.py
from accident_vlm.schemas.preprocessing import PipelineContext


def build_evidence_package(context: PipelineContext) -> dict:
    metadata = context.video_metadata.model_dump() if context.video_metadata else {}
    return {
        "frames": [frame.model_dump() for frame in context.selected_frames],
        "overlays": [],
        "crops": [],
        "precomputed_facts": {
            "metadata": metadata,
            "ocr": context.ocr_observations,
            "tracks": context.tracks,
            "road_geometry": context.road_geometry,
            "speed_estimates": context.speed_and_distance,
            "traffic_control": context.traffic_control,
            "event_candidates": context.event_candidates,
        },
    }
```

- [ ] **Step 4: Implement pre-VLM pipeline assembly**

```python
# src/accident_vlm/pipeline.py
from accident_vlm.config import PipelineConfig
from accident_vlm.modules.evidence_builder import build_evidence_package
from accident_vlm.modules.frame_selection import select_regular_frames
from accident_vlm.schemas.preprocessing import PipelineContext, VideoMetadata


def build_pre_vlm_context(
    video_path: str,
    metadata: VideoMetadata,
    config: PipelineConfig | None = None,
) -> PipelineContext:
    active_config = config or PipelineConfig()
    selected_frames = select_regular_frames(
        duration_sec=metadata.duration_sec,
        fps=metadata.fps,
        interval_sec=active_config.regular_frame_interval_sec,
    )
    context = PipelineContext(
        video_path=video_path,
        video_metadata=metadata,
        selected_frames=selected_frames,
    )
    context.evidence_package = build_evidence_package(context)
    return context
```

- [ ] **Step 5: Run test and verify pass**

Run: `pytest tests/test_pipeline_contract.py -v`

Expected: 1 passed.

## Task 8: VLM Composer Interface

**Files:**
- Create: `src/accident_vlm/modules/vlm_composer.py`
- Modify: `src/accident_vlm/pipeline.py`
- Create: `tests/fixtures/sample_precomputed.json`

- [ ] **Step 1: Create sample precomputed fixture**

```json
{
  "metadata": {
    "duration_sec": 6.0,
    "fps": 30,
    "resolution": "1920x1080",
    "frame_count": 180,
    "has_audio": false
  },
  "tracks": [
    {
      "track_id": "B",
      "type": "승용차",
      "relative_position_start": "전방우측",
      "relative_position_end": "전방중앙",
      "movement_candidate": "차로변경_좌",
      "confidence": "medium"
    }
  ],
  "event_candidates": [
    {
      "time": "00:05.800",
      "event_type": "접촉",
      "actors": ["ego_vehicle", "B"],
      "confidence": "medium",
      "signals": ["bbox_overlap", "camera_shake"]
    }
  ]
}
```

- [ ] **Step 2: Implement prompt builder without calling the model**

```python
# src/accident_vlm/modules/vlm_composer.py
from accident_vlm.schemas.preprocessing import PipelineContext


SYSTEM_PROMPT = """You compose objective accident facts from supplied evidence only.
Do not determine fault ratio, legal violation, negligence, offender, or victim.
Unsupported fields must be marked 확인불가.
Every important event must include confidence and evidence."""


def build_vlm_prompt(context: PipelineContext) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Return JSON matching schema_version accident_video_facts.v1.\n"
        f"Evidence package:\n{context.evidence_package}"
    )
```

- [ ] **Step 3: Add composer protocol for future Qwen backend**

Append to `src/accident_vlm/modules/vlm_composer.py`:

```python
from typing import Protocol


class VLMBackend(Protocol):
    def generate_json(self, prompt: str, image_paths: list[str]) -> dict:
        ...


def compose_with_backend(context: PipelineContext, backend: VLMBackend) -> dict:
    prompt = build_vlm_prompt(context)
    image_paths = [
        frame.get("path")
        for frame in context.evidence_package.get("frames", [])
        if frame.get("path")
    ]
    return backend.generate_json(prompt=prompt, image_paths=image_paths)
```

## Task 9: CLI Entrypoint

**Files:**
- Create: `src/accident_vlm/cli.py`

- [ ] **Step 1: Implement CLI for pre-VLM context generation**

```python
# src/accident_vlm/cli.py
import json
from pathlib import Path

import typer
from rich import print

from accident_vlm.modules.ingestion import probe_video
from accident_vlm.pipeline import build_pre_vlm_context

app = typer.Typer()


@app.command()
def analyze(video_path: Path, output_path: Path = Path("outputs/pre_vlm_context.json")) -> None:
    metadata = probe_video(video_path)
    context = build_pre_vlm_context(video_path=str(video_path), metadata=metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(context.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[green]Wrote pre-VLM context:[/green] {output_path}")
```

- [ ] **Step 2: Run CLI help**

Run: `python -m accident_vlm.cli --help`

Expected: Typer help text renders without import errors.

## Task 10: Verification

**Files:**
- Modify only files created in previous tasks if verification finds defects.

- [ ] **Step 1: Run all unit tests**

Run: `pytest -v`

Expected: all tests pass.

- [ ] **Step 2: Run lint**

Run: `ruff check src tests`

Expected: no lint failures.

- [ ] **Step 3: Run schema smoke test in Python**

Run:

```bash
python - <<'PY'
from accident_vlm.schemas.common import Confidence, EvidenceField, Status
from accident_vlm.schemas.final_output import AccidentFactOutput, AccidentType, SceneType

output = AccidentFactOutput(
    scene_type=EvidenceField(
        value=SceneType.ROAD,
        status=Status.OBSERVED,
        confidence=Confidence.MEDIUM,
        source=["visual"],
        evidence=["frame_000001"],
    ),
    rag_hints={"accident_type": AccidentType.VEHICLE_TO_VEHICLE, "scenario_keywords": ["차로변경사고"]},
    objective_summary="자차와 상대 차량의 접촉이 관찰됨.",
)
print(output.model_dump_json(indent=2))
PY
```

Expected: valid JSON prints with `schema_version` equal to `accident_video_facts.v1`.

## Self-Review

Spec coverage:

- Insurance-style fields are represented in `AccidentFactOutput`.
- Evidence, source, status, and confidence are enforced by `EvidenceField`.
- OCR, actor tracking, road geometry, speed, traffic control, event detection, evidence, VLM, and guard modules have explicit files and extension points.
- Legal judgment prevention is covered by `schema_guard.py`.
- RAG interface is represented by `rag_hints`.

Known planned extension after this plan:

- Add concrete OCR implementation with PaddleOCR/EasyOCR.
- Add concrete detector/tracker backend.
- Add concrete lane segmentation and BEV calibration.
- Add Qwen backend adapter after local serving method is selected.

These extensions should be implemented as separate plans using the stable interfaces created here.
