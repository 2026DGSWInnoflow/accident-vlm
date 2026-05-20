# Accident VLM

Accident VLM is a video fact extraction pipeline that produces objective,
evidence-linked JSON for RAG workflows.

The pipeline records observable facts and supporting evidence. It does not
determine legal liability, fault, violations, offenders, or victims.

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

Generate pre-VLM evidence and final Qwen-composed accident facts:

```bash
accident-vlm analyze-full input.mp4 \
  --pre-vlm-output outputs/pre_vlm_context.json \
  --final-output outputs/accident_facts.json \
  --ocr-backend auto \
  --detector bytetrack \
  --detector-model yolov8x.pt \
  --qwen-model Qwen/Qwen3.6-27B \
  --device auto
```

The final JSON is still evidence constrained: unsupported facts must remain
`확인불가`, and legal judgment terms are sanitized before output.

For multi-GPU Qwen serving, make sure every GPU is visible to the API process.
For example, on a 4 x 24GB server:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACCIDENT_VLM_MAX_MEMORY="0:22GiB,1:22GiB,2:22GiB,3:22GiB,cpu:64GiB"
```

By default the VLM receives all evidence images at original resolution. If the
server is memory constrained, these optional guards can be set:

```bash
export ACCIDENT_VLM_MAX_IMAGES=24
export ACCIDENT_VLM_IMAGE_MAX_SIDE=1024
export ACCIDENT_VLM_MAX_NEW_TOKENS=2048
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
  -F "qwen_model_id=Qwen/Qwen3.6-27B" \
  -F "device=auto"
```

Check status and fetch result:

```bash
curl http://localhost:8000/v1/jobs/{job_id}
curl http://localhost:8000/v1/jobs/{job_id}/result
```

Jobs are stored under `outputs/api_jobs/{job_id}` with intermediate and final JSON files.
