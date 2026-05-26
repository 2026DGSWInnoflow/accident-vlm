import subprocess
import sys

import pytest

from accident_vlm.config import PipelineConfig


def test_config_import_does_not_load_pydantic() -> None:
    script = """
import sys
import accident_vlm.config
print("pydantic" in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def test_pipeline_config_preserves_positive_value_validation() -> None:
    with pytest.raises(ValueError, match="regular_frame_interval_sec"):
        PipelineConfig(regular_frame_interval_sec=0)


def test_pipeline_config_preserves_non_negative_value_validation() -> None:
    with pytest.raises(ValueError, match="event_scan_min_score"):
        PipelineConfig(event_scan_min_score=-1)
