# Accident VLM

Accident VLM is the VLM-stage project in a larger accident analysis flow:
`VLM -> RAG (laws, precedents, fault standards) -> final analysis`.

This repository extracts objective, evidence-linked accident facts from video
and produces JSON for the downstream legal RAG and final analysis stages.

The pipeline records observable facts and supporting evidence. It does not
determine legal liability, fault, violations, offenders, or victims; those
questions belong to downstream RAG and final analysis stages outside this
project.

## Architecture

- [한글 아키텍처 문서](docs/architecture_ko.md)

## Server Setup

Install the core package:

```bash
pip install -e .
```

Install optional runtime groups on the server as needed:

```bash
pip install -e ".[ocr,cv,vlm]"
```

## Commands

Generate the full pre-VLM evidence package:

```bash
accident-vlm analyze input.mp4 outputs/pre_vlm_context.json \
  --ocr-backend auto \
  --detector bytetrack \
  --detector-model yolov8x.pt
```

Generate pre-VLM evidence and VLM-stage Qwen-composed accident facts:

```bash
accident-vlm analyze-full input.mp4 \
  --pre-vlm-output outputs/pre_vlm_context.json \
  --final-output outputs/accident_facts.json \
  --ocr-backend auto \
  --detector bytetrack \
  --detector-model yolov8x.pt \
  --qwen-model Qwen/Qwen3.6-35B-A3B \
  --device auto
```

The VLM-stage JSON is still evidence constrained: unsupported facts must remain
`확인불가`, and legal judgment terms are sanitized before output.

The default pipeline is quality-first. A request that does not override options
uses OCR, YOLOv8x + ByteTrack, denser regular frames, motion keyframes, segment
tracking, road geometry/BEV, traffic control detection, and VLM composition.
Current quality defaults:

```text
regular_frame_interval_sec=0.5
max_selected_frames=32
max_motion_keyframes=16
motion_sample_interval_sec=0.25
min_motion_change_score=6.0
pre_event_window_sec=6.0
post_event_window_sec=4.0
segment_tracking_stride_frames=2
max_segment_tracking_frames=180
object_detector_backend=bytetrack
object_detector_model=yolov8x.pt
```

For multi-GPU Qwen serving, make sure every GPU is visible to the API process.
For example, on a 4 x 24GB server:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACCIDENT_VLM_MAX_MEMORY="0:22GiB,1:22GiB,2:22GiB,3:22GiB,cpu:64GiB"
```

By default the VLM path is local Qwen3.6-35B-A3B through the Transformers
backend. Preprocessing may scan higher-frame evidence, then the VLM receives the
top 20 prioritized/event-window evidence images resized to a 640px max side and
caps VLM-stage JSON generation at 512 tokens. CUDA OOM retries fall back to 12
images before compact text-only evidence.

```bash
export ACCIDENT_VLM_QWEN_MODEL_ID="/home/minsung0830/accident-vlm/models/Qwen3.6-35B-A3B"
export ACCIDENT_VLM_MAX_IMAGES=20
export ACCIDENT_VLM_OOM_RETRY_MAX_IMAGES=12
export ACCIDENT_VLM_IMAGE_MAX_SIDE=640
export ACCIDENT_VLM_IMAGE_CHUNK_SIZE=0
export ACCIDENT_VLM_CHUNK_MAX_NEW_TOKENS=192
export ACCIDENT_VLM_FINAL_MAX_NEW_TOKENS=512
export ACCIDENT_VLM_MODEL_DTYPE=bfloat16
export ACCIDENT_VLM_USE_CACHE=0
```

## API Server

Run the API server:

```bash
accident-vlm-api
```

or:

```bash
uvicorn accident_vlm.server.app:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Start a job from a video path already on the server:

```bash
curl -X POST http://localhost:8000/v1/jobs/from-path \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/data/accidents/sample.mp4",
    "options": {
      "mode": "pre_vlm",
      "ocr_backend": "auto",
      "object_detector_backend": "bytetrack",
      "object_detector_model": "yolov8x.pt"
    }
  }'
```

Upload a video and start a full Qwen job:

```bash
curl -X POST http://localhost:8000/v1/jobs/upload \
  -F "file=@sample.mp4" \
  -F "mode=full" \
  -F "ocr_backend=auto" \
  -F "object_detector_backend=bytetrack" \
  -F "object_detector_model=yolov8x.pt" \
  -F "qwen_model_id=Qwen/Qwen3.6-35B-A3B" \
  -F "device=auto"
```

Check status and fetch result:

```bash
curl http://localhost:8000/v1/jobs/{job_id}
curl http://localhost:8000/v1/jobs/{job_id}/result
```

Jobs are stored under `outputs/api_jobs/{job_id}` with intermediate and VLM-stage JSON files.
