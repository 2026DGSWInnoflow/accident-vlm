# Accident Video Fact Extraction Design

## Purpose

This module analyzes dashcam accident video and produces objective, evidence-linked accident facts for the next RAG stage. It does not determine legal liability, fault ratio, violations, negligence, or victim/offender roles.

The downstream RAG module will use this output to retrieve relevant law, precedents, and Korea insurance association fault-ratio reference materials.

## Product Boundary

This module must:

- Preserve insurance-style accident fields such as scene type, road condition, traffic control, actors, movement, collision type, impact location, and uncertain items.
- Extract only facts that are observed, computed, or cautiously inferred from video, OCR, tracking, metadata, or geometry.
- Mark unavailable or unreliable information as unknown instead of filling gaps.
- Produce machine-readable JSON as the primary output.
- Produce an objective Korean summary as a secondary output.
- Provide RAG search hints without using legal judgment terms.

This module must not:

- Decide fault ratio.
- Say which party violated traffic law.
- Use labels such as offender, victim, liable party, negligent party, or at-fault party.
- Convert weak visual cues into certain facts.
- Output speed, signal, location, lane count, or impact location without source and confidence.

## Pipeline Overview

```text
Input Blackbox Video
  -> A. Ingestion & Quality Analysis
  -> B. Metadata / OCR Extraction
  -> C. Frame & Segment Selection
  -> D. Scene Understanding Preprocessing
  -> E. Actor Detection & Tracking
  -> F. Road Geometry / Lane / BEV Calibration
  -> G. Speed & Distance Estimation
  -> H. Traffic Control Analysis
  -> I. Event & Collision Candidate Detection
  -> J. Evidence Package Builder
  -> K. Qwen VLM Fact Composition
  -> L. Schema Validation / Hallucination Guard
  -> M. Final Accident Fact JSON for RAG
```

The design is accuracy-first. Implementation time is secondary.

## Shared Field Contract

Most structured fields should use this shape:

```json
{
  "value": "normalized_value_or_unknown",
  "raw": "original text or short natural-language observation",
  "status": "observed/computed/inferred/unknown",
  "confidence": "high/medium/low/unknown",
  "source": ["visual", "ocr", "tracker", "geometry", "audio", "metadata", "vlm_review"],
  "evidence": ["frame_000123", "track_B", "traffic_light_crop_000123"],
  "note": "optional reason, assumption, or failure explanation"
}
```

Status meanings:

- `observed`: Directly visible or read from reliable metadata/OCR.
- `computed`: Produced by deterministic or geometric computation.
- `inferred`: Derived from multiple weaker signals.
- `unknown`: Not available or not reliable enough.

## A. Ingestion & Quality Analysis

Role: inspect the input video and estimate analysis reliability.

Inputs:

- Video file: mp4, mov, avi, mkv, or supported dashcam format.

Outputs:

```json
{
  "video_metadata": {
    "duration_sec": 18.2,
    "fps": 30,
    "resolution": "1920x1080",
    "frame_count": 546,
    "has_audio": true
  },
  "input_quality": {
    "blur": "low/medium/high",
    "brightness": "dark/normal/overexposed",
    "night_noise": "low/medium/high",
    "camera_shake": "low/medium/high",
    "occlusion": "low/medium/high",
    "analysis_reliability": "high/medium/low"
  }
}
```

Candidate tools:

- `ffprobe` for metadata.
- OpenCV for frame reading.
- Laplacian variance for blur.
- Brightness histogram for exposure.
- Optical flow or feature displacement for shake.

Quality scores must influence downstream confidence.

## B. Metadata / OCR Extraction

Role: read dashcam overlays and visible text.

Targets:

- Date and time.
- Speed.
- GPS coordinates.
- Address or map overlay.
- Channel name.
- Device information.
- Road signs and speed-limit signs.

Approach:

- Detect likely overlay regions at top/bottom of the frame.
- Run OCR on sampled frames and high-quality keyframes.
- Parse date/time, speed, GPS, and sign text using regex and format-specific parsers.
- Apply temporal voting across frames to reduce OCR noise.

Outputs:

```json
{
  "ocr_observations": [
    {
      "time": "00:01.0",
      "text": "2026-05-19 08:31:22 47km/h",
      "parsed": {
        "datetime": "2026-05-19 08:31:22",
        "speed_kmh": 47
      },
      "bbox": [120, 1010, 650, 1060],
      "confidence": 0.86
    }
  ]
}
```

Priority rules:

- OCR or metadata speed has higher priority than vision-estimated speed.
- OCR or metadata timestamp has higher priority than visual time-of-day estimation.
- OCR GPS/address has higher priority than visual place inference.
- If OCR is inconsistent, keep alternatives and lower confidence.

## C. Frame & Segment Selection

Role: select frames and segments for expensive CV analysis and VLM review.

Frame categories:

- `regular_frames`: representative frames at a fixed interval.
- `motion_keyframes`: frames with high visual change.
- `event_frames`: frames around collision, contact, lane change, pedestrian entry, or sudden stop candidates.
- `pre_event_segment`: typically 5 seconds before main event candidate.
- `post_event_segment`: typically 3 seconds after main event candidate.

Outputs:

```json
{
  "selected_segments": [
    {
      "id": "seg_collision_candidate_1",
      "start": "00:02.5",
      "end": "00:08.5",
      "reason": ["camera_shake", "object_distance_drop"]
    }
  ],
  "selected_frames": [
    {"id": "frame_0030", "time": "00:01.0", "purpose": "context"},
    {"id": "frame_0150", "time": "00:05.0", "purpose": "impact_candidate"}
  ]
}
```

## D. Scene Understanding Preprocessing

Role: classify road and accident scene context before VLM composition.

Scene enum:

- `교차로`
- `횡단보도`
- `일반도로`
- `고속도로`
- `자동차전용도로`
- `주차장`
- `골목길`
- `이면도로`
- `회전교차로`
- `터널`
- `교량`
- `램프구간`
- `합류구간`
- `분기구간`
- `확인불가`

Outputs:

```json
{
  "scene_type_candidates": [
    {
      "value": "일반도로",
      "confidence": "medium",
      "evidence": ["frame_0030", "frame_0090"]
    },
    {
      "value": "교차로",
      "confidence": "low",
      "evidence": ["frame_0120"]
    }
  ]
}
```

The final output should select a primary value only when sufficiently supported. Alternatives can be retained.

## E. Actor Detection & Tracking

Role: identify and track accident participants and relevant surrounding actors.

Actor type enum:

- `승용차`
- `SUV`
- `승합차`
- `버스`
- `화물차`
- `택시`
- `이륜차`
- `자전거`
- `전동킥보드`
- `보행자`
- `동물`
- `고정장애물`
- `도로시설물`
- `확인불가`

Actor role enum:

- `내 차량`
- `상대 차량`
- `제3 차량`
- `보행자`
- `자전거`
- `오토바이`
- `도로시설물`
- `확인불가`

Movement enum:

- `직진`
- `좌회전`
- `우회전`
- `유턴`
- `정차`
- `정지`
- `감속`
- `가속`
- `후진`
- `차로변경_좌`
- `차로변경_우`
- `끼어들기`
- `출발`
- `주차`
- `문열림`
- `횡단`
- `역주행`
- `미끄러짐`
- `확인불가`

Outputs:

```json
{
  "tracks": [
    {
      "track_id": "B",
      "type": "승용차",
      "role_candidate": "상대 차량",
      "positions": [
        {"time": "00:03.0", "bbox": [1280, 520, 1540, 760]},
        {"time": "00:04.0", "bbox": [1160, 525, 1450, 770]}
      ],
      "relative_position_start": "전방우측",
      "relative_position_end": "전방중앙",
      "movement_candidate": "차로변경_좌",
      "confidence": "medium"
    }
  ]
}
```

Candidate tools:

- YOLO or RT-DETR family for object detection.
- ByteTrack or BoT-SORT for multi-object tracking.
- Re-identification and temporal smoothing for track continuity.

## F. Road Geometry / Lane / BEV Calibration

Role: estimate lane count, ego lane, road geometry, and real-world scale.

Targets:

- Visible lane count.
- Ego lane.
- Lane markings: solid, dashed, centerline, shoulder.
- Vanishing point.
- Lane width estimate.
- Bird-eye-view transform.
- Road plane coordinate system.

Outputs:

```json
{
  "road_geometry": {
    "visible_lane_count": {
      "value": 3,
      "confidence": "medium"
    },
    "ego_lane": {
      "value": "2차로",
      "confidence": "low",
      "reason": "카메라 위치와 차선 일부 가림"
    },
    "homography": {
      "available": true,
      "method": "lane_width_vanishing_point",
      "assumptions": ["차선 폭 3.2m", "도로 평면"],
      "confidence": "medium"
    }
  }
}
```

Candidate tools:

- Lane segmentation.
- Vanishing point estimation.
- Homography.
- BEV projection.
- Camera calibration estimation.

Distance and speed modules must use this geometry output. If geometry is weak, speed must become range-based or unknown.

## G. Speed & Distance Estimation

Role: estimate ego and actor speed, actor distance, relative motion, and approach trend.

Priority order:

1. OBD/CAN/GPS/metadata speed if provided externally.
2. OCR overlay speed.
3. BEV geometry plus tracked object movement.
4. Relative motion trend only.
5. Unknown.

Outputs:

```json
{
  "speed_estimates": [
    {
      "actor_id": "ego_vehicle",
      "value": "47km/h",
      "numeric_kmh": 47,
      "range_kmh": [45, 49],
      "method": "ocr_overlay",
      "confidence": "high",
      "evidence": ["ocr_00_01_0", "ocr_00_02_0"]
    },
    {
      "actor_id": "B",
      "value": "약 35~50km/h",
      "numeric_kmh": null,
      "range_kmh": [35, 50],
      "method": "bev_tracking_estimate",
      "confidence": "medium",
      "assumptions": ["차선 폭 3.2m", "도로 평면"],
      "evidence": ["track_B", "road_geometry_homography"]
    }
  ],
  "distance_estimates": [
    {
      "time": "00:04.0",
      "actors": ["ego_vehicle", "B"],
      "distance_m_range": [4.5, 7.0],
      "method": "bev_projection",
      "confidence": "low"
    }
  ],
  "relative_motion": [
    {
      "actor_id": "B",
      "relative_speed_trend": "접근/이탈/유지/확인불가",
      "lateral_movement": "좌측/우측/없음/확인불가",
      "confidence": "medium"
    }
  ]
}
```

Rules:

- Do not output exact numeric speed without method and confidence.
- Prefer ranges over false precision.
- If lane geometry or calibration is weak, report relative trend or unknown.
- Speed must never be generated only by VLM.

## H. Traffic Control Analysis

Role: analyze signal lights, road signs, stop lines, crosswalks, and road-control elements.

Signal enum:

- `녹색`
- `황색`
- `적색`
- `좌회전신호`
- `직진금지`
- `점멸`
- `신호등없음`
- `확인불가`

Sign enum:

- `일시정지`
- `양보`
- `어린이보호구역`
- `노인보호구역`
- `제한속도`
- `진입금지`
- `좌회전금지`
- `유턴금지`
- `주정차금지`
- `횡단보도`
- `자전거도로`
- `버스전용차로`
- `공사안내`
- `확인불가`

Outputs:

```json
{
  "traffic_control": {
    "signal": {
      "value": "확인불가",
      "visible": true,
      "method": "traffic_light_crop_classifier",
      "confidence": "low",
      "reason": "신호등 crop 크기가 작고 역광이 있음"
    },
    "signs": [
      {
        "value": "제한속도",
        "raw_text": "30",
        "confidence": "medium",
        "source": "sign_detection_ocr"
      }
    ],
    "crosswalk": {
      "visible": false,
      "confidence": "medium"
    }
  }
}
```

Candidate tools:

- Traffic light detector.
- Traffic light crop classifier.
- Color-threshold analysis as an auxiliary signal.
- Sign detector.
- OCR.
- Temporal consistency checks.

Signal state should not be trusted from a single small crop unless confidence is high.

## I. Event & Collision Candidate Detection

Role: generate candidate timeline events before VLM.

Event type enum:

- `진입`
- `진행`
- `정차`
- `정지`
- `감속`
- `가속`
- `차로변경시작`
- `차로변경중`
- `회전`
- `횡단시작`
- `접근`
- `급접근`
- `회피조작`
- `충돌`
- `접촉`
- `충돌후정지`
- `이탈`
- `확인불가`

Collision candidate signals:

- Bounding-box overlap or near-overlap.
- Rapid distance decrease.
- Optical-flow spike.
- Camera-shake spike.
- Track discontinuity.
- Visible deformation or occlusion at contact point.
- Post-event speed change.
- Audio impact spike.

Outputs:

```json
{
  "event_candidates": [
    {
      "time": "00:03.2",
      "event_type": "차로변경시작",
      "actors": ["B"],
      "confidence": "medium",
      "signals": ["lane_boundary_crossing", "leftward_track_motion"]
    },
    {
      "time": "00:05.8",
      "event_type": "충돌",
      "actors": ["ego_vehicle", "B"],
      "confidence": "high",
      "signals": ["camera_shake", "bbox_overlap", "audio_spike"]
    }
  ]
}
```

This module proposes candidates. The final JSON should still carry confidence and evidence.

## J. Evidence Package Builder

Role: create compact multimodal input for Qwen VLM.

Package contents:

- Original keyframes.
- Before/during/after event frames.
- Actor crops.
- Traffic-light crops.
- Sign crops.
- Lane overlay images.
- BEV overlay images.
- Tracking overlay images.
- Precomputed JSON observations.

Outputs:

```json
{
  "evidence_package": {
    "frames": [
      "frame_0090_before.jpg",
      "frame_0150_impact_candidate.jpg",
      "frame_0180_after.jpg"
    ],
    "overlays": [
      "tracking_overlay_00_03_to_00_06.jpg",
      "bev_overlay_00_03_to_00_06.jpg"
    ],
    "crops": [
      "actor_B_00_04.jpg",
      "traffic_light_00_03.jpg"
    ],
    "precomputed_facts": {
      "metadata": {},
      "ocr": [],
      "tracks": [],
      "road_geometry": {},
      "speed_estimates": [],
      "traffic_control": {},
      "event_candidates": []
    }
  }
}
```

The VLM should receive evidence, not an unconstrained full-video prompt.

## K. Qwen VLM Fact Composition

Role: compose the final accident facts from images, crops, overlays, and precomputed data.

Model target:

- `Qwen/Qwen3.6-27B` or a compatible Qwen multimodal serving setup.

Prompt requirements:

- Use only the supplied evidence package.
- Record observable facts only.
- Do not decide legal liability, fault ratio, violation, negligence, offender, or victim.
- Mark unsupported fields as `확인불가`.
- If preprocessing and image evidence conflict, record the conflict in `uncertainties`.
- Every important timeline event must include confidence and evidence.
- Never invent speed, signal state, exact location, lane count, or impact location.

VLM responsibilities:

- Select final scene type from candidates.
- Compose actors from tracks and visual review.
- Write timeline events objectively.
- Summarize collision facts without legal judgment.
- Generate RAG hints from normalized facts.
- Write a short objective Korean summary.

## L. Schema Validation / Hallucination Guard

Role: validate, normalize, and reject unsupported VLM output.

Checks:

- Required fields exist.
- Values are in enum or explicitly unknown.
- Actor IDs match tracks or declared actors.
- Timeline is sorted by time.
- Important claims have confidence and evidence.
- Speed has method and confidence.
- Signal state has source and confidence.
- Collision exists only when supported by event evidence.
- Legal judgment terms are absent.
- Contradictions are moved to `uncertainties`.

Forbidden legal judgment terms:

- `가해`
- `피해`
- `과실`
- `위반`
- `불법`
- `책임`
- `주의의무`
- `신호위반`
- `안전거리 미확보`

If these terms appear outside explicit "not_determined" metadata, the output must be regenerated or sanitized.

## M. Final Accident Fact JSON

Primary output:

```json
{
  "schema_version": "accident_video_facts.v1",
  "input_quality": {},
  "scene_type": {},
  "road_conditions": {
    "weather": {},
    "surface": {},
    "time": {},
    "visibility": {}
  },
  "traffic_control": {
    "signal": {},
    "signs": [],
    "crosswalk": {}
  },
  "actors": [],
  "timeline": [],
  "collision": {},
  "speed_and_distance": {},
  "uncertainties": [],
  "evidence_index": {},
  "rag_hints": {
    "accident_type": {},
    "scenario_keywords": [],
    "required_followup_questions": []
  },
  "objective_summary": ""
}
```

Example objective summary:

```text
자차는 일반도로에서 직진 주행 중이었고, 전방 우측 차량이 좌측 방향으로 이동하면서 두 차량 간 거리가 감소하였다. 00:05.8 부근에서 자차 우측 전방과 상대 차량 좌측 부근의 접촉이 관찰된다. 신호등 색상과 제한속도는 영상에서 명확히 확인되지 않는다.
```

## Final Output Enums

### Road Conditions

Weather:

- `맑음`
- `흐림`
- `비`
- `눈`
- `안개`
- `강한역광`
- `확인불가`

Surface:

- `건조`
- `젖음`
- `적설`
- `결빙`
- `공사구간`
- `노면불량`
- `확인불가`

Time:

- `주간`
- `야간`
- `새벽_해질녘`
- `확인불가`

Visibility:

- `양호`
- `불량`
- `부분불량`
- `확인불가`

### Lane Or Position

Use structured position instead of a single enum:

```json
{
  "lane": "1차로/2차로/3차로/갓길/중앙선부근/차로외/확인불가",
  "relative_position": "전방/후방/좌측/우측/전방좌측/전방우측/후방좌측/후방우측/확인불가",
  "area": "횡단보도/교차로내/교차로진입전/교차로통과후/주차면/인도/확인불가"
}
```

### Impact Type

- `추돌`
- `정면충돌`
- `측면충돌`
- `접촉`
- `보행자충돌`
- `자전거충돌`
- `이륜차충돌`
- `시설물충돌`
- `단독사고`
- `비접촉`
- `충돌없음`
- `확인불가`

### Impact Location

Separate ego and other actor:

```json
{
  "ego_vehicle": "전면/후면/좌측/우측/좌측전방/우측전방/좌측후방/우측후방/하부/확인불가",
  "other_actor": "전면/후면/좌측/우측/좌측전방/우측전방/좌측후방/우측후방/확인불가"
}
```

### Damage Or Injury

- `부상있음`
- `부상없음`
- `물피`
- `차량손상관찰`
- `시설물손상관찰`
- `확인불가`

### RAG Accident Type

- `차대차`
- `차대보행자`
- `차대자전거`
- `차대이륜차`
- `차대전동킥보드`
- `차대시설물`
- `단독사고`
- `비접촉사고`
- `다중추돌`
- `확인불가`

### RAG Scenario Keywords

Base keyword enum:

- `교차로사고`
- `신호등있는교차로`
- `신호등없는교차로`
- `차로변경사고`
- `진로변경사고`
- `추돌사고`
- `후진사고`
- `주차장사고`
- `개문사고`
- `횡단보도사고`
- `보행자횡단사고`
- `좌회전중사고`
- `우회전중사고`
- `유턴중사고`
- `합류중사고`
- `고속도로사고`
- `이면도로사고`
- `중앙선침범의심`
- `비접촉사고`
- `동일방향주행`
- `대향방향주행`
- `선행차량`
- `후행차량`
- `직진차량`
- `정차차량`
- `주정차차량`

RAG hints may add extra non-legal descriptive keywords when they are directly grounded in the final facts.

## Confidence Policy

Confidence levels:

- `high`: Supported by direct observation or multiple consistent modules.
- `medium`: Supported by one reliable module or several weaker signals.
- `low`: Possible but visually or computationally weak.
- `unknown`: Not available or not reliable enough.

Confidence must be lowered when:

- Video quality is poor.
- Object is small, occluded, or blurred.
- Geometry calibration is weak.
- OCR is inconsistent across frames.
- VLM observation conflicts with preprocessing.
- The value depends on a strong assumption such as lane width.

## RAG Interface

The downstream RAG module should receive:

- Normalized accident type.
- Scene type.
- Road context.
- Actor types and movements.
- Collision type and impact location.
- Timeline events.
- Confirmed and uncertain traffic-control facts.
- Missing follow-up questions.
- Objective summary.

The RAG module should not infer that a low-confidence field is confirmed. It should use uncertainty to broaden or qualify retrieval.

## Implementation Notes

Recommended module boundaries:

- `video_ingestion`
- `ocr_extraction`
- `frame_selection`
- `scene_preprocessing`
- `actor_tracking`
- `road_geometry`
- `speed_distance`
- `traffic_control`
- `event_detection`
- `evidence_builder`
- `vlm_fact_composer`
- `schema_guard`

Each module should expose structured input/output models and should be testable independently.

## Open Implementation Decisions

These are implementation choices, not product ambiguities:

- Exact object detector and tracker.
- Exact lane segmentation model.
- Exact OCR model and fallback strategy.
- Qwen serving backend.
- JSON schema library.
- Storage format for evidence images and overlays.

The product contract above should remain stable while these choices are evaluated.
