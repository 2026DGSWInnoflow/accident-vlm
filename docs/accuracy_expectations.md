# Accuracy Expectations

This project extracts objective accident facts. The numbers below are planning
estimates, not benchmark claims. They should be replaced by `accident-vlm
evaluate-dataset` results once enough labeled videos have been processed.

## Current Quality Target: High-Precision Mode

The default target is now high precision: facts that remain in the final JSON
should be correct at an estimated 90%+ rate. This is achieved by lowering recall:
ambiguous facts are explicitly downgraded to `확인불가` instead of being guessed.

For 100 mixed dashcam accident videos with visible road context, expected
precision of retained facts:

- Broad scene / actor presence retained facts: 90-94%
- Accident type retained hints: 90-93%
- Actor timeline retained events: 90-92%
- Traffic light / sign retained facts: 90-92%
- OCR speed / datetime / GPS retained facts: 90-94% when overlay text is present
- Collision retained facts: 90-92%
- RAG-ready objective JSON precision: 90-93%

Expected recall is lower:

- RAG-ready JSON with enough concrete facts: 55-70 out of 100
- Conservative JSON with many `확인불가` fields: 30-45 out of 100

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
- High-precision gate downgrades low-confidence timeline, actor movement, traffic
  signal, speed, collision, and RAG accident type hints unless they have strong
  preprocessing evidence.
