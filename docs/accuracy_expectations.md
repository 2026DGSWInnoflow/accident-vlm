# Accuracy Expectations

This project extracts objective accident facts. The numbers below are planning
estimates, not benchmark claims. They should be replaced by `accident-vlm
evaluate-dataset` results once enough labeled videos have been processed.

## Current Quality Target

For 100 mixed dashcam accident videos with visible road context:

- Broad scene / actor presence: 70-85 usable outputs
- Accident type hint: 65-80 usable outputs
- Actor timeline: 60-75 usable outputs
- Traffic light / sign facts: 50-70 usable outputs
- OCR speed / datetime / GPS: 45-65 usable outputs when overlay text is present
- Collision candidate timing: 55-70 usable outputs
- RAG-ready objective JSON: 65-80 usable outputs

## Main Failure Modes

- Night video, glare, rain, heavy blur, or blocked camera view
- Missing or unreadable dashcam overlay text
- Small distant traffic lights or signs
- Dense traffic where generic YOLO tracking fragments actors
- Collision outside visible frame or after the clip ends

## New Quality Controls

- Evidence images now receive `importance_score` and are ranked before VLM input.
- Track fragments are consolidated before event detection.
- Event candidates receive `event_score` for segment mining and prioritization.
- Traffic light votes include margin diagnostics and confidence based on temporal votes.
- OCR speed summaries reject implausible temporal jumps.
- VLM output is verified against preprocessing evidence before schema validation.
- Dataset evaluation estimates usable/high-quality counts per 100 labeled videos.
