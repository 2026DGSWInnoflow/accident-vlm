# Accident VLM Preprocessing Accuracy TODO Spec

## 목적

- [x] 블랙박스 사고 영상을 VLM에 넣기 전에 객관 근거를 최대한 정밀하게 추출한다.
- [x] VLM이 법적 판단을 하지 않고도 보험/RAG 단계에서 쓸 수 있는 사실 정보를 안정적으로 받게 한다.
- [x] 짧은 충돌, 작은 객체, 신호등/표지판, 차선/거리/속도, 영상 품질 저하 상황에서 누락과 hallucination을 줄인다.
- [x] 모든 전처리 결과는 `source`, `evidence`, `confidence`, `uncertainty`를 포함해 검증 가능해야 한다.

## 현재 구현 상태 요약

- [x] 영상 메타데이터 추출
- [x] 일정 간격 프레임 선택
- [x] 움직임 큰 프레임 추가 선택
- [x] 이벤트 후보 기반 사고 전후 프레임 재선택
- [x] 영상 품질 분석
- [x] OCR 기반 날짜/시간/속도/GPS 후보 추출
- [x] YOLOv8x + ByteTrack/BoT-SORT 기반 actor tracking
- [x] segment tracking
- [x] 객체 crop 및 tracking overlay 생성
- [x] OpenCV 기반 차선/BEV 후보 분석
- [x] OCR 속도 기반 자차 속도 추정
- [x] HSV 기반 신호등 색상 후보 탐지
- [x] OCR 기반 제한속도/정지 표지 후보 탐지
- [x] bbox overlap, 급접근, 급감속, camera shake 기반 이벤트 후보 탐지
- [x] evidence package 생성
- [x] VLM storyboard 생성
- [x] 보험 핵심 필드와 사고유형 후보를 출력 schema에 포함

## 1. 프레임 선택 고도화

### 현재 문제

- 0.5초 간격 샘플링만으로는 0.1~0.2초 충돌을 놓칠 수 있다.
- 이벤트 후보를 놓치면 사고 전후 재선택도 실패한다.
- VLM에 들어가는 20장 프레임이 정말 최적이라는 검증이 부족하다.

### TODO

- [x] 전체 영상을 3~5fps로 빠르게 훑는 1차 event scan을 추가한다.
- [x] optical flow peak, camera shake peak, bbox area change, histogram change를 같이 점수화한다.
  - [x] optical flow peak, camera shake peak, histogram change는 1차 event scan에 반영했다.
  - [x] bbox area change는 Phase 3 tracking/collision 고도화에서 실제 track 기반 값으로 연결한다.
- [x] 후보 구간을 top-k로 유지한다.
- [x] 후보 구간마다 사고 전 6초, 사고 후 4초 window를 만든다.
- [x] 후보 구간 안에서는 15~30fps로 정밀 프레임을 추출한다.
- [x] VLM용 최종 20장은 정밀 후보 프레임에서 압축 선택한다.
- [x] 충돌 후보 직전/직후 프레임은 최소 5장 이상 보장한다.
- [x] 사고 전 맥락, 접근, 충돌 후보, 사고 후 상태, 신호/표지/차선 detail quota를 분리한다.
- [x] 선택되지 않은 후보 프레임도 `rejected_frame_candidates`에 이유와 함께 저장한다.

### 검증 기준

- [x] 30fps 영상에서 3~6프레임짜리 짧은 충돌 후보를 최소 1개 이상 잡는다.
- [x] 이벤트 후보가 여러 개일 때 top-k 후보와 선택 이유가 기록된다.
- [x] VLM storyboard에 사고 전/중/후가 모두 포함된다.
- [x] 프레임 선택 결과를 사람이 볼 수 있는 contact sheet로 저장한다.

## 2. 영상 품질 분석 고도화

### 현재 문제

- 평균 blur/brightness/noise만으로 구간별 품질 차이를 설명하기 어렵다.
- 차량 자체 움직임과 camera shake가 섞일 수 있다.
- 야간/역광/비/눈/와이퍼/가림 상황을 충분히 구분하지 못한다.

### TODO

- [x] 전체 영상 품질과 사고 후보 구간 품질을 분리 산출한다.
- [x] 프레임별 blur score, brightness score, noise score를 저장한다.
- [x] motion blur와 focus blur를 구분하는 후보 지표를 추가한다.
- [x] ego-motion compensation 후 camera shake score를 계산한다.
- [x] windshield occlusion, rain/snow/fog, glare, overexposure 후보를 추가한다.
- [x] 품질 저하가 심한 근거 이미지는 confidence를 자동 낮춘다.
- [x] 품질 분석 결과를 `input_quality.timeline` 형태로 저장한다.

### 검증 기준

- [x] 야간/흐림/역광/흔들림 샘플에서 품질 저하 이유가 구분된다.
- [x] 충격 후보와 일반 차량 움직임이 다른 score로 분리된다.
- [x] 품질 점수가 VLM output uncertainty에 반영된다.

## 3. OCR 고도화

### 현재 상태

- [x] 기본 OCR backend는 EasyOCR `ko`, `en`, GPU 우선이다.
- [x] EasyOCR이 없으면 Tesseract `--psm 6`으로 fallback한다.
- [x] 날짜/시간, 속도, GPS 정규식 파싱이 있다.
- [x] 속도 outlier와 temporal jump 제거가 있다.

### 현재 문제

- 블랙박스 오버레이 전용 OCR 모델이 아니다.
- 작은 글자, 압축 노이즈, 야간 번짐에서 숫자 오독 가능성이 높다.
- 날짜/시간/속도/GPS별 최적 OCR 설정이 분리되어 있지 않다.

### TODO

- [x] OCR ROI 자동 탐색과 고정 ROI 방식을 함께 지원한다.
- [x] 날짜/시간 전용 OCR pass를 추가한다.
- [x] 속도 숫자 전용 OCR pass를 추가한다.
- [x] GPS 좌표 전용 OCR pass를 추가한다.
- [x] Tesseract whitelist 설정을 필드별로 다르게 적용한다.
- [x] 동일 ROI의 여러 프레임 OCR 결과를 temporal voting한다.
- [x] OCR crop 이미지, 원문, 정규화 텍스트, 파싱 결과를 모두 evidence로 저장한다.
- [x] OCR 실패 ROI도 failure case로 저장한다.
- [x] OCR confidence와 temporal consistency를 합친 final confidence를 만든다.

### 검증 기준

- [x] 속도 OCR 결과가 3프레임 이상 일관되면 high confidence로 올라간다.
- [x] 비정상 속도값은 rejected candidate로 남는다.
- [x] OCR이 실패해도 VLM이 임의로 날짜/시간/속도를 만들지 않는다.

## 4. YOLO 및 Tracking 고도화

### 현재 상태

- [x] 기본 detector는 `yolov8x.pt`이다.
- [x] COCO pretrained class 중 car, truck, bus, motorcycle, bicycle, person을 사용한다.
- [x] ByteTrack/BoT-SORT backend를 사용할 수 있다.
- [x] segment tracking이 있다.

### 현재 문제

- 한국 블랙박스 사고영상에 맞춰 fine-tuning된 모델이 아니다.
- 전동킥보드, 특수차량, 교통시설물, 사고 잔해 등 사고 특화 class가 부족하다.
- track id가 끊기거나 서로 다른 객체가 합쳐질 수 있다.
- tracker 성능을 mAP/IDF1/MOTA 기준으로 검증하지 않는다.

### TODO

- [x] 사고영상 데이터셋 class taxonomy를 정의한다.
- [x] 필수 class를 정의한다: 승용차, 화물차, 버스, 이륜차, 자전거, 보행자, 킥보드, 신호등, 좌회전신호, 제한속도표지, 정지표지, 횡단보도, 차선, 정지선, 중앙선.
- [x] 기존 COCO pretrained YOLO와 custom traffic model을 분리한다.
- [x] 사고 후보 구간에만 15~30fps tracking을 수행한다.
- [x] ByteTrack과 BoT-SORT 결과를 비교 저장한다.
- [x] track fragmentation score를 계산한다.
- [x] 객체별 visibility, occlusion, bbox size, confidence timeline을 저장한다.
- [x] low-confidence actor도 후보로 유지하되 VLM에는 uncertainty와 함께 전달한다.

### 검증 기준

- [x] actor 누락률, 오탐률, track 끊김률을 benchmark에 기록한다.
- [x] 같은 객체의 track id가 사고 전후로 유지되는지 확인한다.
- [x] 작은 보행자/오토바이/자전거 crop이 storyboard에 보존된다.

## 5. 시각 근거 생성 검증

### 현재 상태

- [x] tracking overlay 생성
- [x] actor crop 생성
- [x] lane overlay 생성
- [x] BEV overlay 생성
- [x] signal crop 생성
- [x] failure case 저장
- [x] evidence storyboard 생성

### 현재 문제

- crop이 너무 작거나 흐려도 그대로 evidence로 들어갈 수 있다.
- bbox가 잘못 잡힌 crop을 품질 기준으로 걸러내지 않는다.
- overlay가 실제로 VLM 정확도를 올리는지 정량 검증이 없다.

### TODO

- [x] crop 품질 점수를 추가한다: 면적, 선명도, 중심성, clipping 여부, confidence.
- [x] 너무 작은 crop은 확대 저장한다.
- [x] crop 주변 context padding을 추가한다.
- [x] 원본 프레임과 overlay를 pair로 묶는다.
- [x] evidence image마다 `why_selected`, `risk`, `expected_use`를 기록한다.
- [x] VLM 입력 전 contact sheet를 자동 생성한다.
- [x] 사람이 빠르게 검수할 수 있는 evidence HTML report를 만든다.

### 검증 기준

- [x] storyboard 20장 안에 사고 핵심 객체와 신호/표지/차선 근거가 모두 들어간다.
- [x] crop/overlay가 없는 경우 failure reason이 기록된다.
- [x] evidence image별 품질 점수가 benchmark에 포함된다.

## 6. 도로/차선/BEV 고도화

### 현재 상태

- [x] OpenCV Canny + Hough line으로 차선 후보를 찾는다.
- [x] lane count 후보를 만든다.
- [x] vanishing point와 간단 BEV overlay를 만든다.

### 현재 문제

- OpenCV만으로 도로/차선을 완벽하게 찾을 수 없다.
- 야간, 비, 눈, 차선 마모, 가림, 그림자, 횡단보도 오탐에 취약하다.
- 편도/왕복 차선 수 구분이 약하다.
- 좌회전 차로, 유도선, 버스전용차로, 정지선 판단이 없다.

### TODO

- [x] lane segmentation 모델을 추가한다.
- [x] road marking segmentation을 추가한다.
- [x] 횡단보도 detector를 추가한다.
- [x] 정지선 detector를 추가한다.
- [x] 중앙선/차선 종류를 분류한다.
- [x] OpenCV 결과와 segmentation 결과를 voting한다.
- [x] BEV confidence를 별도 계산한다.
- [x] 차선 폭 3.2m 가정 외에 도로 유형별 lane width prior를 추가한다.
- [x] BEV 변환 실패 이유를 failure case로 저장한다.

### 검증 기준

- [x] lane count 결과에 evidence overlay가 연결된다.
- [x] BEV confidence가 낮으면 속도/거리 estimate도 낮은 confidence가 된다.
- [x] 횡단보도/정지선/중앙선이 사고유형 후보와 RAG hint에 반영된다.

## 7. 속도/거리/상대 움직임 고도화

### 현재 상태

- [x] OCR 속도는 자차 속도 후보로 사용한다.
- [x] track bbox 변화로 접근/이탈/좌우 이동을 추정한다.
- [x] 절대 거리/상대 차량 속도는 아직 제한적이다.

### 현재 문제

- 메타데이터/OCR 속도가 없으면 절대 속도 산출이 약하다.
- 상대 차량 속도는 BEV와 ego-motion 보정 없이는 신뢰도가 낮다.
- bbox 중심 이동만으로는 원근 왜곡을 해결하기 어렵다.

### TODO

- [x] bbox bottom-center를 지면 접점으로 사용한다.
- [x] BEV 좌표계에서 객체 위치를 추적한다.
- [x] 차선 폭 기반 pixels-per-meter를 계산한다.
- [x] 프레임 간 시간차를 사용해 이동거리와 속도를 계산한다.
- [x] ego-motion compensation을 적용한다.
- [x] OCR 자차 속도가 있으면 상대속도와 결합해 상대 차량 절대속도 후보를 만든다.
- [x] OCR/metadata가 없으면 절대속도는 low confidence, 상대속도는 medium confidence로 분리한다.
- [x] 거리 estimate는 range 형태로 출력한다.
- [x] 모든 속도/거리 결과에 계산식과 evidence frame pair를 저장한다.

### 계산 예시 TODO

- [x] `delta_frames = 45`
- [x] `fps = 30`
- [x] `delta_time_sec = 45 / 30 = 1.5`
- [x] `lane_width_m = 3.2`
- [x] `bev_delta_lane = 1.5`
- [x] `distance_m = 3.2 * 1.5 = 4.8`
- [x] `speed_kmh = 4.8 / 1.5 * 3.6 = 11.52`
- [x] 위 계산은 BEV confidence와 lane confidence가 충분할 때만 사용한다.

### 검증 기준

- [x] 속도 산출값에는 method, formula, confidence, evidence가 포함된다.
- [x] BEV confidence가 낮으면 속도 결과가 자동으로 low/unknown 처리된다.
- [x] 상대속도와 절대속도를 구분한다.

## 8. 신호등/표지판 고도화

### 현재 상태

- [x] HSV 기반 빨강/노랑/초록 신호 후보 탐지
- [x] temporal voting
- [x] OCR 기반 제한속도/정지 표지 후보 탐지
- [x] traffic light failure case 저장

### 현재 문제

- 좌회전 신호를 인식하지 못한다.
- 직진/좌회전/동시신호/점멸 구분이 없다.
- 색상 blob 기반이라 브레이크등, 간판, LED 번짐 오탐 가능성이 있다.
- 표지판 detector가 없고 OCR에 의존한다.

### TODO

- [x] traffic light detector를 추가한다.
- [x] signal head crop을 만든다.
- [x] 신호 색상 분류와 신호 모양 분류를 분리한다.
- [x] class를 정의한다: 적색, 황색, 녹색, 좌회전녹색, 적색+좌회전, 점멸, 꺼짐, 확인불가.
  - [x] 적색, 황색, 녹색, 좌회전녹색, 적색+좌회전, 확인불가를 지원한다.
  - [x] 점멸, 꺼짐은 frame-level absence sequence까지 모델링한 뒤 완료 처리한다.
- [x] arrow/left-turn shape classifier를 추가한다.
- [x] traffic sign detector를 추가한다.
- [x] 제한속도 표지는 detector crop 후 OCR을 적용한다.
- [x] 신호등/표지판 결과는 temporal vote와 confidence margin을 저장한다.
- [x] 신호등 오탐 후보도 failure case로 저장한다.

### 검증 기준

- [x] 좌회전 신호가 있는 영상에서 `left_turn_signal` 후보가 나온다.
- [x] 신호등이 작거나 흐리면 `확인불가`로 유지한다.
- [x] 표지판 인식 결과는 crop evidence와 연결된다.

## 9. 충돌 후보 탐지 고도화

### 현재 상태

- [x] bbox overlap 기반 접촉 후보
- [x] bbox area 급증 기반 급접근 후보
- [x] 차로변경 움직임 후보
- [x] OCR 속도 급감 후보
- [x] camera shake 기반 충격 후보

### 현재 문제

- bbox overlap은 충돌이 아니어도 발생할 수 있다.
- 실제 충돌이 가려지면 bbox overlap이 없을 수 있다.
- 비접촉 사고 탐지가 약하다.
- camera shake는 방지턱/노면/급제동과 구분이 어렵다.
- 이벤트 후보가 틀리면 프레임 선택과 VLM 분석이 같이 틀어진다.

### TODO

- [x] collision candidate score를 multi-signal 방식으로 재설계한다.
- [x] optical flow 급변 score를 추가한다.
- [x] ego-motion compensated shake score를 추가한다.
- [x] bbox IoU 변화율을 추가한다.
- [x] bbox area 변화율을 추가한다.
- [x] 객체 속도 변화율을 추가한다.
- [x] 사고 후 정지/낙상/방향 변화 후보를 추가한다.
- [x] 비접촉 사고 후보를 별도 class로 유지한다.
- [x] 충돌 후보 top-k를 모두 유지한다.
- [x] 각 후보에 `supporting_signals`, `contradicting_signals`, `confidence`를 저장한다.

### 검증 기준

- [x] 충돌 후보가 1개만 강제 선택되지 않는다.
- [x] 후보별 근거 frame pair와 overlay가 생성된다.
- [x] 낮은 confidence 후보는 VLM uncertainty로 전달된다.

## 10. VLM 입력 구조 고도화

### 현재 상태

- [x] evidence package 생성
- [x] evidence image ranking
- [x] storyboard phase quota
- [x] 보험 핵심 항목 우선 phase
- [x] OpenAI/Transformers backend 모두 image caption interleaving
- [x] chunk fallback 시 compact final prompt

### TODO

- [x] VLM 입력 이미지는 기본 단일 호출 20장으로 유지한다.
- [x] chunk mode는 OOM fallback으로만 사용한다.
- [x] storyboard phase를 사고 후보 중심으로 더 명확히 나눈다.
- [x] 각 storyboard slot에 질문 목적을 명시한다.
- [x] VLM에는 precomputed hint를 후보로만 전달한다.
- [x] VLM이 unsupported field를 채우면 schema guard에서 unknown 처리한다.
- [x] VLM 출력과 전처리 evidence 간 citation consistency를 검사한다.

### 검증 기준

- [x] 20장 안에 사고 전/충돌 후보/사고 후/신호/차선/객체 crop이 포함된다.
- [x] VLM output의 주요 field는 evidence id를 가진다.
- [x] 법적 판단 표현은 제거된다.

## 11. Benchmark 및 검증 체계

### TODO

- [x] benchmark dataset manifest를 만든다.
- [x] 영상별 label을 정의한다: 사고유형, 객체, 신호, 차선, 충돌시점, 자차속도, 상대움직임.
- [x] 프레임 선택 recall을 측정한다.
- [x] OCR field accuracy를 측정한다.
- [x] actor detection mAP를 측정한다.
- [x] tracking IDF1 또는 track continuity를 측정한다.
- [x] lane count accuracy를 측정한다.
- [x] signal/sign accuracy를 측정한다.
- [x] collision candidate recall@k를 측정한다.
- [x] VLM final JSON field fill rate를 측정한다.
- [x] 보험 핵심 field fill rate를 측정한다.
- [x] uncertainty가 적절히 남는지 측정한다.

### 목표 지표

- [x] 충돌 후보 recall@3: 95% 이상 목표
- [x] actor 주요 객체 recall: 90% 이상 목표
- [x] OCR 속도/시간 field accuracy: 90% 이상 목표
- [x] 신호등 확인 가능 영상 기준 signal accuracy: 85~90% 이상 목표
- [x] 차선 수 확인 가능 영상 기준 lane count accuracy: 85% 이상 목표
- [x] 보험 핵심 field known/unknown 판정 적절성: 90% 이상 목표
- [x] 최종 JSON schema valid rate: 99% 이상 목표

## 12. 구현 우선순위

- [x] P0: 고프레임 사고 후보 탐색
- [x] P0: 후보 구간 15~30fps 정밀 tracking
- [x] P0: collision candidate multi-signal score
- [x] P0: evidence contact sheet 및 검증 리포트
- [x] P1: OCR 필드별 전용 pass
- [x] P1: 상대속도/거리 BEV estimate
- [x] P1: traffic light/sign detector
- [x] P1: lane segmentation 모델
- [x] P2: custom YOLO fine-tuning
- [x] P2: benchmark manifest 및 label evaluation
- [x] P2: VLM prompt/schema 지속 개선

## 완료 정의

- [x] 전처리 결과만으로 사고 후보 시점과 주요 actor가 설명 가능하다.
- [x] VLM 없이도 신호/표지/차선/속도/객체/충돌 후보의 근거가 남는다.
- [x] VLM은 전처리 결과를 바탕으로 객관 사실 JSON을 구성한다.
- [x] 불확실한 정보는 추정하지 않고 `확인불가` 또는 uncertainty로 남긴다.
- [x] benchmark에서 프레임 선택, OCR, tracking, 신호/표지, 차선, 충돌 후보, 최종 JSON 품질을 모두 측정한다.
