# 검증 설계 및 구현 계획

이 문서는 **검증 설계 준비와 분할 구현**만 다룬다. 모델 구조 결정, 학습, 모델 성능 검증, 튜닝, 앙상블·보정, 추론·제출은 이번 단계에서 실행하지 않는다.

## 1. 결정 사항

- 무작위 행 분할과 `GroupKFold`를 사용하지 않는다.
- 2025년 평가를 흉내 내도록 과거 시즌만 학습 영역에 두고 바로 다음 시즌 전체를 outer validation으로 둔다.
- canonical 설계는 2022·2023·2024의 expanding 3-fold다.
- 2024 fold는 2025에 가장 가까운 **최근성 앵커**다. 기존 EDA와 과거 실행에서 이미 정답을 확인했으므로 완전히 untouched인 holdout이라고 부르지 않는다.
- 2023 fold는 `game_type=F` 정의·분포 단절을 견디는지 확인하는 **체제 변화 스트레스**다.
- 2022 fold는 단절 이전 구간의 **안정성 검사**다.
- 2023만 사용해 2024를 보는 recent-only split은 오래된 시즌 민감도를 살피는 보조 진단이며 canonical 평균에 섞지 않는다.
- outer validation은 early stopping, checkpoint, 전처리 fit에 절대 사용하지 않는다.

## 2. canonical outer/inner 시간 분할

| 이름 | 역할 | Inner train → inner stop | Outer refit 영역 → outer validation |
| --- | --- | --- | --- |
| `rolling_origin_2022` | 안정성 | 2019–2020 → 2021 | 2019–2021 → 2022 |
| `rolling_origin_2023` | 체제 변화 스트레스 | 2019–2021 → 2022 | 2019–2022 → 2023 |
| `rolling_origin_2024` | 최근성 앵커 | 2019–2022 → 2023 | 2019–2023 → 2024 |
| `recent_only_2023_to_2024` | 보조 학습창 민감도 | canonical에서 확정된 절차 재사용 | 2023 → 2024 |

Inner split은 나중에 epoch 수나 중단 시점을 정할 때만 쓰는 계약이다. 이번 단계에서는 행 위치와 무결성만 구현하며 어떠한 학습도 실행하지 않는다. Outer train 전체를 사용하는 작업이 필요할 때는 inner 영역에서 미리 확정된 절차를 사용해야 하며, outer validation을 보면서 선택을 다시 하면 안 된다. 2024는 이미 여러 번 관찰된 이력이 있으므로 이후에는 새 설정을 2022·2023에서 먼저 고정한 뒤 2024를 확인하는 운영 규칙으로 추가 과적합을 줄인다.

## 3. 실제 데이터 분할 감사

| Outer fold | Train 행 | Validation 행 | Train 타깃률 | Validation 타깃률 | 차이 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2019–2021 → 2022 | 728,588 | 247,472 | 54.314% | 52.892% | -1.422%p |
| 2019–2022 → 2023 | 976,060 | 245,525 | 53.954% | 49.996% | -3.958%p |
| 2019–2023 → 2024 | 1,221,585 | 253,507 | 53.158% | 48.611% | -4.548%p |
| 2023 → 2024 보조 | 245,525 | 253,507 | 49.996% | 48.611% | -1.385%p |

각 분할에서 train/validation `row_id` 교집합은 0이고, train 최대 시즌은 validation 최소 시즌보다 작다. 파일의 `row_id` 숫자 순서도 train이 validation보다 엄격히 앞선다.

`game_type=F`의 train→validation 성공률 변화는 2022 fold +4.59%p, 2023 fold -20.32%p, 2024 fold -17.70%p다. 따라서 전체 평균 하나로 fold를 합치면 체제 단절이 숨겨질 수 있다.

2024 validation에서 train에 이미 등장한 행 비율은 투수 80.14%, 타자 90.70%, 둘 다 72.78%다. 2023만 학습하는 보조 split에서는 둘 다 등장한 비율이 66.61%로 떨어지므로 recent-only 결과는 데이터 기간 효과와 OOV 효과가 함께 바뀌는 민감도 진단으로 해석해야 한다.

## 4. 분할별 fit 경계

다음 상태는 매 outer 또는 inner train 영역에서만 새로 fit한다.

- 범주 사전과 빈도 필터
- missing/OOV 인덱스 외의 범주 인덱스
- 수치 중앙값, 평균, 표준편차, clip 기준
- smoothing prior와 타깃 기반 룩업이 추가되는 경우 그 모든 통계
- target encoding이 추가되는 경우 train 내부에서 다시 분리한 값

Validation에서는 이미 fit된 객체로 `transform`만 호출한다. Validation 행의 빈도, 평균, ID 출현 횟수, 행 순서나 다른 validation 행을 전처리 상태에 반영하지 않는다.

공식 `asof_*` 입력은 각 행에 제공된 값을 그대로 사용한다. 같은 선수의 다음 행을 조회하거나 차분·역방향 채움·양방향 시퀀스를 만들지 않는다.

TrackMan을 사용할 경우 대상 행의 시즌보다 엄격히 이전인 `source_season`만 누적된 target-season lookup을 사용한다. 메인·TrackMan 선수 ID를 직접 연결하지 않는다.

## 5. 성능 보고 계약

아직 모델 성능은 계산하지 않았지만 이후 검증 결과의 형식은 지금 고정한다.

- 1차 지표: fold별 전체 Brier score
- 파생 지표: fold 자체 타깃률 기준 Brier skill과 공식 점수식
- 보조 지표: log loss, ECE
- 필수 slice: `game_type` F/R
- 필수 entity slice: both seen, pitcher only, batter only, neither seen
- 필수 history slice: 최근 경기 결측/관측, 낮은 `asof_pitcher_n`, 투수·타자 cold-start

2024 최근성 앵커와 2023 체제 변화 fold를 각각 별도로 보고한다. 3-fold OOF 평균은 보조 요약으로만 사용하며, 한 평균값이 특정 fold의 실패를 가리지 않게 한다.

## 6. 구현·검증 항목

- frozen fold 명세와 deterministic position 생성
- outer train/validation 및 inner train/stop의 strict season order 검사
- 빈 영역, 중복 `row_id`, train/validation 교집합, 예상하지 않은 시즌 즉시 실패
- row 위치와 원본 순서 보존
- fold-train에서만 전처리 `fit`을 호출하는 helper 계약
- canonical 3-fold와 recent-only 보조 split의 manifest 생성
- fold, 경기유형, entity coverage, 결측 slice 진단 CSV 생성
- 합성 데이터 단위 테스트와 전체 1,475,092행 manifest 감사

여기서 구현을 종료한다. 모델을 생성하거나 학습·예측을 실행하지 않는다.

## 7. 실행 완료 산출물

- `src/lg_control/validation.py`: immutable 시간 분할, inner stop, 선택 범위 차단, train-only 전처리 fit helper
- `tests/test_validation.py`: 시간 순서·행 보존·누수 차단·오류 입력 계약 테스트 12개
- `scripts/build_validation_design.py`: 전체 데이터에서 동일한 manifest와 감사표를 재생성하는 진입점
- `artifacts/validation_design/manifest.json`: 입력 SHA-256, 분할별 row ID sequence hash, 정책과 향후 보고 계약
- `artifacts/validation_design/season_summary.csv`: 시즌별 데이터 분포
- `artifacts/validation_design/fold_summary.csv`: outer/inner 행 수, 타깃·경기 유형 변화, entity coverage와 결측률
- `artifacts/validation_design/subgroup_summary.csv`: 향후 검증에서 반드시 볼 고정 slice의 행 수·기초 타깃률
- `artifacts/validation_design/temporal_integrity.csv`: 시간 순서, 행·ID 비중복, inner/outer 격리 결과

전체 1,475,092행에서 manifest 생성을 완료했고, 검증 설계 신규 테스트 12개를 포함한 회귀 테스트 30개가 통과했다. 이 산출물에는 모델, 예측값, 예측 성능 지표, checkpoint가 없다.
