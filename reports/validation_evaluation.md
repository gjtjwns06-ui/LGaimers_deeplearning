# Development outer-validation 예측 및 성능 평가

실행일: 2026-08-23

이 보고서는 이미 학습이 끝난 `rolling_origin_2022`, `rolling_origin_2023` checkpoint의 outer validation 예측과 성능만 다룬다. 모델 재학습, epoch/checkpoint 재선택, hyperparameter tuning, calibration, ensemble, 2024 anchor 예측·채점, test 추론·제출은 수행하지 않았다.

## 1. 평가 계약

- 1차 지표: fold별 Brier score(낮을수록 좋음)
- 기준선: 해당 fold의 실제 성공률 `r`을 모든 행에 예측한 Brier `r(1-r)`
- 파생 지표: `1 - Brier / baseline`인 Brier skill과 `max(0, 100000 × skill)`인 공식식 환산 점수
- 보조 지표: log loss, equal-width 15-bin ECE
- 예측 전 validation target 열을 제거하고, checkpoint 전처리기와 모델에는 feature와 row ID만 전달했다.
- 예측이 원본 outer-validation row 순서와 정확히 정렬된 뒤에만 target을 결합해 지표를 계산했다.

## 2. Fold별 1차 결과

| Fold | 행 | Target rate | Mean prediction | Baseline Brier | Model Brier | Brier skill | 공식식 환산 점수 | Log loss | ECE-15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2019–2021 → **2022** | 247,472 | 0.528920 | 0.533650 | 0.249164 | **0.244325** | **+0.019420** | **1,942.04** | 0.681559 | 0.009711 |
| 2019–2022 → **2023** | 245,525 | 0.499957 | 0.524028 | 0.250000 | **0.255005** | **-0.020020** | **0.00** | 0.704377 | 0.035078 |

2022에서는 상수 기준선보다 Brier가 0.004839 낮아 유효한 일반화 신호가 확인됐다. 반면 2023에서는 기준선보다 0.005005 높아졌고, 평균 예측도 실제 성공률보다 약 2.41%p 높았다. 따라서 이 모델은 시기 변화에 안정적으로 일반화됐다고 볼 수 없다.

여기의 점수는 대회 공식식과 같은 산식을 validation fold에 적용한 **환산값**이며 실제 비공개 test 점수가 아니다.

## 3. 사전 고정 slice 진단

가장 큰 실패는 검증 설계 단계에서 체제 변화 스트레스로 지정했던 2023 `game_type=F`에서 나타났다.

| Fold / Slice | 행 비중 | Target rate | Mean prediction | Brier | Skill | ECE-15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 / `game_type=F` | 12.30% | 0.708749 | 0.659859 | 0.211210 | -0.023185 | 0.051696 |
| 2022 / `game_type=R` | 87.70% | 0.503691 | 0.515943 | 0.248971 | +0.004063 | 0.012323 |
| 2023 / `game_type=F` | 10.46% | 0.472904 | **0.705886** | **0.303368** | **-0.217046** | **0.232982** |
| 2023 / `game_type=R` | 89.54% | 0.503118 | 0.502779 | 0.249354 | +0.002544 | 0.012018 |

2023 `F`에서 실제 성공률은 47.29%인데 모델은 평균 70.59%를 예측했다. 반면 같은 시즌 `R`은 평균과 calibration이 비교적 잘 맞고 기준선보다 소폭 낫다. 2023 전체 실패는 특히 `F` 정의·분포 단절과 강하게 연결돼 있다. 이는 slice 간 연관을 보여주는 진단이며 단독 인과 추정은 아니다.

추가 관찰:

- 2023 entity-status 네 그룹은 모두 음의 skill이다. `both_seen`도 -0.016683이므로 단순 OOV 선수 문제만으로 2023 실패를 설명할 수 없다.
- 2023 recent-history 결측 2,722행은 target 0.4897 대비 평균 0.5884를 예측했고 Brier 0.271251, ECE 0.104292였다.
- 2023 투수 low-history 3,066행과 타자 low-history 3,030행의 skill은 각각 -0.106466, -0.160061이었다.
- cold-start slice는 fold별 63~88행 수준으로 매우 작으므로 방향성 이상의 해석을 하지 않는다.
- `history_missing` slice는 두 fold 모두 0행이며 계약에 따라 삭제하지 않고 metric을 null로 저장했다.

전체 고정 slice 결과는 `artifacts/validation_evaluation/slice_metrics.csv`에 있다.

## 4. 보조 aggregate

2022·2023을 이어 붙인 development OOF는 492,997행이다.

| 요약 | Brier | Brier skill | 공식식 환산 점수 | Log loss | ECE-15 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fold macro mean | 0.249665 | -0.000300 | 971.02 | 0.692968 | 0.022395 |
| Row-pooled OOF | 0.249644 | +0.000585 | 58.48 | 0.692923 | 0.018750 |

Pooled 수치는 서로 반대인 2022 개선과 2023 실패를 거의 상쇄한다. 특히 점수의 0 절단은 비선형이므로 fold 점수 평균을 공식 점수처럼 해석하지 않는다. 이 프로젝트의 primary 결과는 위의 fold별 Brier다.

## 5. 산출물 및 무결성

- `artifacts/validation_evaluation/rolling_origin_2022/predictions.csv`
- `artifacts/validation_evaluation/rolling_origin_2023/predictions.csv`
- `artifacts/validation_evaluation/development_oof_predictions.csv`
- `artifacts/validation_evaluation/fold_metrics.csv`
- `artifacts/validation_evaluation/slice_metrics.csv`
- `artifacts/validation_evaluation/aggregate_metrics.json`
- fold별 `evaluation_record.json`
- `artifacts/validation_evaluation/manifest.json`

Strict verifier가 다음 항목을 다시 확인했다.

- 정확한 8개 tracked-file inventory와 모든 SHA-256
- fold별 247,472 / 245,525행, 원본 row ID 순서와 hash
- 확률의 finite 및 `[0,1]` 범위
- fold, pooled, macro, row-weighted, worst-fold 지표 재계산
- 원본 데이터와 모든 고정 slice 지표 재계산
- OOF가 2022 → 2023 fold prediction의 정확한 결합인지 여부
- 평가 전후 learning artifact의 byte-level 동일성
- 2024 anchor 예측·채점 및 tuning/calibration/ensemble/inference/submission 미실행

전체 회귀 및 평가 계약 테스트는 55개가 통과했다.
