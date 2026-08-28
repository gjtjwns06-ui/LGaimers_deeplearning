# 2023–2024 신체제 딥러닝 실행 결과

실행일: 2026-08-27

## 결론

`game_type=F`의 과거 성공률 shortcut이 깨진 2023년 이후를 기준으로 새 딥러닝
파이프라인을 구현하고 최종 3-seed 모델과 제출 ZIP까지 생성했다.

- 2019–2022 F 원본 행/타깃의 전처리 fit, class prior, main loss, auxiliary loss 노출: **0행**
- 보조로 유지한 과거 데이터: 체제가 비교적 안정적인 2019–2022 R만 사용
- 새 체제 학습: 2023–2024 F/R 전체 사용
- 2024 동일가중 OOF: Brier **0.2482119086**, 환산 **638.50점**
- 정직한 기존 nested DL 0.248354545 대비: **-0.000142636 개선**
- 기존 2024 반복 탐색 최고 DL 0.247987554 대비: **+0.000224354 악화**

따라서 새 모델은 최근 체제 기준 연구 gate는 통과했지만 기존 최고 모델을 대체하지는
못했다. `submit_recent_regime.zip`은 신체제 전용 대안 제출물이며 최고 점수 제출물로
표시하지 않는다.

## 학습 데이터 계약

최종 base fit은 총 1,369,784행이다.

| Season | F | R |
|---:|---:|---:|
| 2019 | **제외** | 211,627 |
| 2020 | **제외** | 220,874 |
| 2021 | **제외** | 221,227 |
| 2022 | **제외** | 217,024 |
| 2023 | 25,686 | 219,839 |
| 2024 | 30,010 | 223,497 |

과거 R은 시즌당 `0.6^age`로 감쇠하고, 2023–2024는 동일 가중으로 사용했다. 이후
2023–2024의 499,032행만으로 작은 학습률의 고정 4-epoch polish를 수행했다.

엄밀한 보장 범위는 “pre-2023 F 원본 행과 타깃을 fit/loss/prior에 직접 노출하지
않음”이다. 2023–2024 행에 공식 입력으로 제공된 `asof_*` 자체에는 그 선수의 과거
누적 이력이 포함될 수 있으므로, 모든 역사 정보가 0이라고 주장하지 않는다.

## 모델과 선택 절차

- row-local rate-basis 전처리
- compact DCNv2-style cross mixture-of-experts
- `success`, `reverse_only`, `middle_only`, `reverse_and_middle`,
  `other_failure`의 exhaustive 5-class head
- main Brier 직접 최적화와 ball/strike auxiliary loss
- seed: 42, 2026, 3407, 동일가중 확률 평균
- calibration 없음
- test 행 간 집계·빈도·rolling·차분 없음

epoch와 학습창은 2024가 아니라 2023 월별 inner split으로 결정했다.

| Inner 후보 | Fit | 2023 late Brier | F bias | 선택 |
|---|---|---:|---:|---|
| recent-only | 2023 월 4–6 | 0.249296087 | +3.88%p | 탈락 |
| stable-R + recent | 과거 R + 2023 월 4–6 | **0.248618883** | **+0.41%p** | 선택 |

base epoch는 2, recent-only polish epoch는 4로 고정했다. 2024는 이미 이 프로젝트에서
반복 관찰한 앵커이므로 아래 수치를 untouched/blind 성능이라고 부르지 않는다.

## 2024 확인 결과

### Seed별

| Seed | Brier | 환산 점수 |
|---:|---:|---:|
| 42 | 0.2482741740 | 613.57 |
| 2026 | 0.2483307744 | 590.92 |
| 3407 | 0.2482248064 | 633.34 |
| **동일가중** | **0.2482119086** | **638.50** |

세 seed 모두 기존 nested equal DL Brier 0.248354545보다 낮아 방향이 일치했다.

### Slice

| Slice | 실제 평균 | 예측 평균 | Bias | Brier |
|---|---:|---:|---:|---:|
| 전체 | 0.486105 | 0.490992 | +0.004887 | **0.248211909** |
| F | 0.459280 | 0.458860 | -0.000420 | **0.247907497** |
| R | 0.489707 | 0.495306 | +0.005599 | **0.248252783** |

F의 과거 양의 uplift는 제거됐고 F 평균은 정확하게 맞았다. 남은 약점은 R 전체의
+0.56%p 과대예측과 7–10월 성능 저하다.

## 최종 fresh 학습

검증 checkpoint를 warm-start하지 않고 각 seed를 새로 초기화했다.

1. 과거 R + 2023–2024 F/R로 base 2 epoch
2. 같은 모델을 2023–2024만으로 고정 4 epoch polish
3. 세 seed 동일가중

최종 번들의 정책 감사 결과:

- fit rows: 1,369,784
- recent polish rows: 499,032
- obsolete pre-2023 F rows: 0
- model count: 3
- row-independent inference: true

## 검증

- 전체 테스트: **210 passed**
- 5행 ZIP 실행 smoke: 통과
- 단일 행/전체 batch 최대 차이: `5.96e-08`
- 셔플/원순서 복원 최대 차이: `5.96e-08`
- 다른 행 입력 변조가 기준 행 예측에 준 차이: `0.0`
- CPU 245,789행 모사 추론: 약 **13.9초**
- ZIP 크기: 1,218,512 bytes
- ZIP SHA-256: `3ec8ae3faa22eebebd41bd4fa176e637eea1438ecf194ab0329a6e4cf7b48eea`

## 산출물

- `artifacts/final_recent_regime/`
- `artifacts/recent_regime_outer_ensemble_predictions.csv`
- `submit_recent_regime.zip`
- `submit_recent_regime.zip.sha256`
- `src/lg_control/recent_regime.py`
- `src/lg_control/recent_regime_final.py`
- `scripts/train_recent_regime_final.py`
- `scripts/package_recent_regime_submission.py`

최종 manifest SHA-256:
`2650c5ccd43463d81d4f1d1ff5beb5f0ab0102e0f378a8bd2a460e4b351fc959`
