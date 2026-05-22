# Preprocessing Accuracy Phase Plan

이 문서는 `docs/preprocessing_accuracy_todo_spec.md`의 TODO를 한 번에 섞어 수정하지 않기 위한 실행 계획이다.
각 phase는 독립 검증을 통과한 뒤 다음 phase로 넘어간다.

## 공통 원칙

- [x] 더미 데이터나 placeholder 구현으로 TODO를 완료 처리하지 않는다.
- [x] 실제 영상 처리 경로에 연결되지 않은 코드는 완료로 보지 않는다.
- [x] 각 산출물은 `source`, `evidence`, `confidence`, `uncertainty` 또는 실패 이유를 남긴다.
- [x] 각 phase 종료 시 phase 내부 리뷰와 전체 파이프라인 호환성 리뷰를 수행한다.
- [x] 체크박스는 구현, 테스트, 파이프라인 연결, 문서 반영이 모두 끝난 항목만 체크한다.

## Phase 0: 범위 감사 및 실행 순서 고정

- [x] 기존 전처리 문서와 구현 상태를 확인한다.
- [x] TODO 간 의존성을 정리한다.
- [x] phase 순서를 확정한다.
- [x] 전체 TODO 문서의 체크박스는 실제 구현 완료 시점에만 갱신한다.

### Phase 0 리뷰

- [x] 프레임/이벤트 후보 탐색이 후속 tracking, collision, VLM storyboard의 선행 조건임을 확인했다.
- [x] 속도/거리 고도화는 BEV, tracking, ego-motion 고도화 뒤에 수행해야 함을 확인했다.
- [x] 신호/표지/차선 detector는 기존 HSV/OCR/OpenCV 결과와 voting 구조로 붙이는 것이 안전함을 확인했다.

## Phase 1: 고프레임 사고 후보 탐색 및 프레임 선택 재구성

- [x] 전체 영상 3~5fps 1차 event scan을 추가한다.
- [x] optical flow peak, camera shake peak, histogram change를 점수화한다.
- [x] 가능한 경우 bbox area change를 score 입력으로 받을 수 있게 구조화한다.
- [x] top-k event scan 후보를 `context.event_scan_candidates`에 저장한다.
- [x] 후보별 pre/post window를 생성한다.
- [x] 후보 구간 안에서 15~30fps 정밀 프레임 후보를 생성한다.
- [x] VLM용 최종 20장 압축 선택 로직에 event scan 후보를 반영한다.
- [x] rejected frame candidates와 거절 이유를 저장한다.
- [x] contact sheet를 생성한다.
- [x] Phase 1 테스트, lint, 전체 호환성 리뷰를 완료한다.

### Phase 1 리뷰

- [x] `event_scan.py`가 실제 영상을 OpenCV로 샘플링하여 frame diff, optical flow, camera shake peak, histogram change를 계산한다.
- [x] 짧은 충돌 대비를 위해 3~5fps 1차 scan과 후보 구간 15~30fps 정밀 프레임 선택을 분리했다.
- [x] event scan 후보는 `event_scan_candidates`, 선택 제외 프레임은 `rejected_frame_candidates`, 검수 이미지는 `contact_sheets`로 evidence package에 연결된다.
- [x] event scan 목적값을 evidence scoring, VLM storyboard phase, VLM image priority에 연결했다.
- [x] 기존 regular/motion/event detection 흐름은 유지하고 event scan 후보를 추가 후보로 병합하여 호환성 충돌을 줄였다.
- [x] 검증: `ruff check src tests`, `PYTHONPATH=src pytest -q` 통과.

## Phase 2: 영상 품질 분석 고도화와 evidence 품질 점수

- [x] 프레임별 blur, brightness, noise, motion score timeline을 저장한다.
- [x] 전체 영상 품질과 사고 후보 구간 품질을 분리한다.
- [x] ego-motion compensated shake score를 추가한다.
- [x] glare, overexposure, rain/fog/snow 후보 지표를 추가한다.
- [x] evidence image 품질 점수를 산출하고 confidence 보정에 연결한다.
- [x] Phase 2 테스트, lint, 전체 호환성 리뷰를 완료한다.

### Phase 2 리뷰

- [x] `InputQuality.timeline`에 프레임별 blur, brightness, noise, motion, compensated motion, glare, dark, contrast, blur type을 저장한다.
- [x] `InputQuality.segment_quality`는 event scan window 기준 품질 reliability를 별도 산출한다.
- [x] camera shake score에 affine 기반 ego-motion compensated value를 추가했다.
- [x] visibility 후보로 glare, overexposure, low light, rain/snow/fog/dirty lens, windshield occlusion을 기록한다.
- [x] evidence image마다 실제 파일을 읽어 품질 점수와 quality confidence를 부여하고 낮은 품질은 ranking penalty를 받는다.
- [x] 전체 호환성 리뷰: 기존 `InputQuality` 필드에는 default를 추가했으므로 기존 schema 사용 코드는 깨지지 않는다.
- [x] 검증: `ruff check src tests`, `PYTHONPATH=src pytest -q` 통과.

## Phase 3: Tracking 및 충돌 후보 multi-signal 고도화

- [x] 후보 구간 15~30fps tracking을 파이프라인 기본 흐름에 연결한다.
- [x] track fragmentation, visibility, occlusion, bbox size timeline을 저장한다.
- [x] collision candidate score를 multi-signal 방식으로 재설계한다.
- [x] optical flow 급변, bbox IoU 변화율, bbox area 변화율, 객체 속도 변화율을 추가한다.
- [x] 비접촉 사고 후보와 사고 후 정지/낙상/방향 변화 후보를 추가한다.
- [x] 후보별 supporting/contradicting signals를 저장한다.
- [x] Phase 3 테스트, lint, 전체 호환성 리뷰를 완료한다.

### Phase 3 리뷰

- [x] segment tracking 결과와 일반 tracking 결과를 `consolidate_tracks`에서 병합한 뒤 track quality를 산출한다.
- [x] segment frame id의 여러 숫자 그룹을 잘못 합치지 않고 마지막 frame index를 기준으로 gap과 span을 계산한다.
- [x] track별 fragmentation score, visibility, occlusion candidate, bbox area timeline을 저장한다.
- [x] 접촉 후보는 bbox overlap 단독이 아니라 bbox IoU, IoU 변화율, camera shake, optical flow event scan, bbox area 변화, 객체 속도 변화를 함께 근거로 남긴다.
- [x] 비접촉후보, 사고후정지후보, 방향변화후보, 낙상후보를 별도 candidate class로 유지한다.
- [x] 전체 호환성 리뷰: event scan 후보는 기존 후보와 병합되고, 새 신호는 `supporting_signals`/`contradicting_signals` 필드로 추가되어 기존 schema를 깨지 않는다.
- [x] 검증: `ruff check src/accident_vlm/modules/event_detection.py src/accident_vlm/modules/track_consolidation.py src/accident_vlm/pipeline.py tests/test_event_detection.py tests/test_pipeline_contract.py`, `PYTHONPATH=src pytest -q tests/test_event_detection.py tests/test_pipeline_contract.py::test_consolidate_tracks_adds_track_quality_without_misreading_segment_ids tests/test_pipeline_contract.py::test_analyze_video_pre_vlm_connects_event_scan_candidates` 통과.

## Phase 4: OCR 필드별 고도화

- [x] OCR ROI 자동 탐색과 고정 ROI를 함께 사용한다.
- [x] 날짜/시간, 속도, GPS 전용 OCR pass를 추가한다.
- [x] Tesseract whitelist를 필드별로 분리한다.
- [x] temporal voting과 final confidence를 강화한다.
- [x] OCR crop, 원문, 정규화 텍스트, 파싱 결과, 실패 ROI를 evidence로 저장한다.
- [x] Phase 4 테스트, lint, 전체 호환성 리뷰를 완료한다.

### Phase 4 리뷰

- [x] 고정 ROI와 top/bottom text contour 기반 자동 ROI를 함께 생성한다.
- [x] `datetime`, `speed`, `gps`, `general` OCR pass를 분리하고 ROI crop을 field별로 저장한다.
- [x] Tesseract fallback은 field별 whitelist와 psm 설정을 사용한다.
- [x] OCR observation에 crop path, 원문, normalized text, parsed field, target field, 실패 이유를 기록한다.
- [x] summary에는 field vote, failure count, temporal consistency, final confidence score가 포함된다.
- [x] 전체 호환성 리뷰: 기존 backend는 `field_hint`를 받지 않아도 fallback 호출로 동작하고, 기존 observation schema는 유지된다.
- [x] 검증: `ruff check src tests`, `PYTHONPATH=src pytest -q tests/test_ocr_summary.py tests/test_speed_distance.py tests/test_pipeline_contract.py tests/test_vlm_composer.py tests/test_schema_guard.py` 통과.

## Phase 5: 도로/차선/BEV 및 상대 속도/거리 고도화

- [x] lane segmentation backend interface를 추가한다.
- [x] OpenCV lane 결과와 segmentation 결과를 voting한다.
- [x] 횡단보도, 정지선, 중앙선, road marking 후보를 추가한다.
- [x] BEV confidence와 failure reason을 산출한다.
- [x] bbox bottom-center 기반 BEV 위치 추적을 추가한다.
- [x] 차선 폭, fps, frame delta 기반 속도/거리 계산식을 저장한다.
- [x] OCR 자차 속도와 상대속도를 결합하는 보정 경로를 추가한다.
- [x] Phase 5 테스트, lint, 전체 호환성 리뷰를 완료한다.

### Phase 5 리뷰

- [x] OpenCV Hough lane 결과를 lane detection vote 구조로 감싸고 segmentation backend는 optional slot으로 유지한다.
- [x] 정지선, 횡단보도, 황색 중앙선 후보를 road marking candidate로 저장한다.
- [x] BEV confidence score, homography failure reason, lane width prior를 별도 필드로 기록한다.
- [x] bbox bottom-center를 BEV 좌표로 변환해 거리, 속도, range, 계산식, evidence frame pair를 저장한다.
- [x] OCR 자차 속도가 있으면 상대 BEV 속도와 함께 absolute speed range 후보를 생성한다.
- [x] 전체 호환성 리뷰: 기존 `visible_lane_count`, `lane_segmentation.method`, `homography` 필드는 유지하고 새 필드는 추가만 했다.
- [x] 검증: `ruff check src/accident_vlm/modules/road_geometry.py src/accident_vlm/modules/speed_distance.py tests/test_road_geometry.py tests/test_speed_distance.py`, `PYTHONPATH=src pytest -q tests/test_road_geometry.py tests/test_speed_distance.py tests/test_pipeline_contract.py` 통과.

## Phase 6: 신호등/표지판 detector 및 좌회전 신호 고도화

- [x] traffic light detector interface를 추가한다.
- [x] signal head crop을 생성한다.
- [x] 색상 분류와 모양/화살표 분류를 분리한다.
- [x] 좌회전녹색, 적색+좌회전, 점멸, 꺼짐, 확인불가 class를 지원한다.
  - [x] 좌회전녹색, 적색+좌회전, 확인불가 class를 지원한다.
  - [x] 점멸/꺼짐은 frame-level absence sequence까지 모델링한 뒤 완료 처리한다.
- [x] traffic sign detector interface를 추가한다.
- [x] 제한속도 표지는 detector crop 후 OCR을 적용한다.
- [x] temporal vote, confidence margin, 오탐 failure case를 저장한다.
- [x] Phase 6 테스트, lint, 전체 호환성 리뷰를 완료한다.

### Phase 6 리뷰

- [x] 신호 후보 crop은 signal head crop으로 저장되고 HSV 색상과 arrow shape가 분리 기록된다.
- [x] 녹색 좌화살표 crop은 `좌회전녹색`, 적색과 좌회전녹색이 함께 보이면 `적색+좌회전`으로 voting된다.
- [x] 신호/표지판 vote에는 confidence margin과 diagnostics가 포함된다.
- [x] 원형 적색 표지판 후보는 `sign_crop`으로 저장하고 OCR 표지 결과와 함께 evidence package로 들어간다.
- [x] 분류 불가 신호 후보와 신호 미검출은 failure case로 저장된다.
- [x] 전체 호환성 리뷰: 기존 `signal.value`, `signs`, `sign_votes`, `failure_cases`는 유지하고 shape/crops/diagnostics/sign_crops를 추가했다.
- [x] 검증: `ruff check src/accident_vlm/modules/traffic_control.py tests/test_traffic_control.py`, `PYTHONPATH=src pytest -q tests/test_traffic_control.py` 통과.

## Phase 7: VLM 입력, citation consistency, benchmark 완성

- [x] VLM 기본 입력은 단일 호출 20장을 유지하고 chunk는 OOM fallback으로 제한한다.
- [x] storyboard phase와 slot 질문 목적을 사고 후보 중심으로 강화한다.
- [x] VLM output과 preprocessing evidence citation consistency를 검사한다.
- [x] benchmark manifest schema를 만든다.
- [x] frame recall, OCR accuracy, actor recall, tracking continuity, lane count accuracy, signal accuracy, collision recall@k를 측정한다.
- [x] 보험 핵심 field fill rate와 uncertainty 적절성을 측정한다.
- [x] 목표 지표를 report에 포함한다.
- [x] Phase 7 테스트, lint, 전체 호환성 리뷰를 완료한다.

### Phase 7 리뷰

- [x] benchmark manifest schema `accident_vlm.benchmark_manifest.v1`을 추가하고 video path/label 필수 구조를 검증한다.
- [x] collision recall@3, frame recall, actor recall, OCR speed accuracy, tracking continuity, lane count accuracy, signal accuracy를 계산한다.
- [x] 보험 핵심 field fill/known-unknown 적절성과 uncertainty presence를 측정한다.
- [x] VLM evidence consistency는 schema guard와 fact verifier에서 evidence 없는 주요 field를 unknown으로 강등한다.
- [x] 전체 호환성 리뷰: benchmark API runner 기존 출력 구조를 유지하고 manifest/metric 함수는 독립 함수로 추가했다.
- [x] 검증: `ruff check src/accident_vlm/benchmark.py tests/test_benchmark.py tests/test_evaluation.py`, `PYTHONPATH=src pytest -q tests/test_benchmark.py tests/test_evaluation.py` 통과.

## Phase 8: Tracker 비교, event overlay, preprocessing uncertainty 연결

- [x] ByteTrack/BoT-SORT 비교 결과를 저장한다.
- [x] actor 누락률, 오탐률, track 끊김률 benchmark metric을 추가한다.
- [x] 충돌 후보별 근거 frame pair overlay를 생성한다.
- [x] low-confidence actor/event와 입력 품질 저하를 VLM preprocessing uncertainty로 전달한다.
- [x] 작은 actor crop은 padding, 확대, 품질/risk와 함께 storyboard 후보로 보존한다.
- [x] Phase 8 테스트, lint, 전체 호환성 리뷰를 완료한다.

### Phase 8 리뷰

- [x] `tracker_comparison`이 evidence package precomputed facts에 저장된다.
- [x] `event_candidate_overlay`가 evidence image ranking에 들어가며 event candidate에 overlay id가 역참조된다.
- [x] `preprocessing_uncertainties`가 compact VLM facts에 포함되어 낮은 신뢰도 근거가 최종 uncertainty로 이어질 수 있다.
- [x] 전체 호환성 리뷰: tracker comparison은 config option으로 추가되어 기존 기본 실행 비용을 바꾸지 않고, event overlay는 기존 overlay list에 append된다.
- [x] 검증: `ruff check ...`, `PYTHONPATH=src pytest -q tests/test_evidence_visuals.py tests/test_pipeline_contract.py tests/test_actor_tracking.py tests/test_vlm_composer.py`, `PYTHONPATH=src pytest -q tests/test_benchmark.py` 통과.

## Phase 9: Lane segmentation model, custom YOLO training/evaluation

- [x] ONNX lane segmentation backend를 추가한다.
- [x] OpenCV Hough lane 결과와 segmentation lane count를 voting한다.
- [x] accident-specific YOLO dataset YAML 생성기를 추가한다.
- [x] Ultralytics custom YOLO fine-tuning 실행 helper를 추가한다.
- [x] actor detection mAP/mAP50 평가 helper를 추가한다.
- [x] Phase 9 테스트, lint, 전체 호환성 리뷰를 완료한다.

### Phase 9 리뷰

- [x] lane segmentation model path가 제공되면 OpenCV DNN ONNX backend로 실제 mask inference를 수행한다.
- [x] segmentation overlay/mask는 evidence image로 들어갈 수 있는 `lane_segmentation_overlay` purpose를 유지한다.
- [x] custom YOLO training/eval helper는 Ultralytics API를 직접 호출하며 taxonomy YAML을 사고 특화 class 기준으로 생성한다.
- [x] 전체 호환성 리뷰: 모델 경로가 없으면 기존 OpenCV-only 경로가 그대로 유지되고, 학습 helper는 런타임 pipeline과 분리되어 서버 분석 속도에 영향이 없다.
- [x] 검증: `PYTHONPATH=src pytest -q tests/test_training.py tests/test_road_geometry.py`, `ruff check src/accident_vlm/training.py src/accident_vlm/modules/road_geometry.py tests/test_training.py tests/test_road_geometry.py` 통과.

## 최종 완료 조건

- [x] `preprocessing_accuracy_todo_spec.md`의 모든 TODO가 실제 구현과 테스트를 근거로 체크되어 있다.
- [x] 전체 테스트가 통과한다.
- [x] 전체 phase 리뷰가 문서에 남아 있다.
- [x] 새 전처리 결과가 evidence package와 VLM storyboard에 연결되어 있다.
