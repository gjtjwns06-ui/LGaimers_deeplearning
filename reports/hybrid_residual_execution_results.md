# 외부 v7 + 딥러닝 잔차 결합 실행 결과

실행일: 2026-08-27

## 결론

Public 937.308점의 `open1-v7`을 frozen base로 두고, 현재 딥러닝 두 모델의
시간순 OOF·보조 head·행 로컬 원본 피처만으로 bounded logit residual MLP를
구현했다. 2021에서 학습하고 2022에서 선택한 모델은 개발 게이트를 통과했지만,
2023 체제 변화에서 악화했고 2024 최종 확인에서도 v7보다 악화했다.

따라서 이 residual family는 **최종 FAIL**이며, full-train 재학습·통합 추론·ZIP은
생성하지 않았다.

## 누수 없는 nested OOF

outer season `t`의 epoch를 `t-1`에서 선택하고, `season < t`를 선택된 epoch만큼
다시 학습했다. outer target은 metric 계산에만 사용했으며 checkpoint 선택에는
사용하지 않았다. cosine scheduler horizon은 원래 탐색과 동일하게 4epoch로 고정했다.

| Outer | rate-exclusive epoch | temporal-TE epoch |
|---:|---:|---:|
| 2021 | 2 | 2 |
| 2022 | 2 | 4 |
| 2023 | 3 | 1 |
| 2024 | 1 | 1 |

2023 붕괴 연도가 두 2024 모델 모두 epoch 1을 선택했다. 기존 2024 단일 holdout에서
outer label로 선택했던 rate-exclusive epoch 4와 달리, nested rate 모델의 2024
점수는 451.80에 그쳤다.

## frozen v7 대비 OOF 기준선

| Season | v7 score | nested DL equal score | 50:50 score | 사후 최적 DL 비중 |
|---:|---:|---:|---:|---:|
| 2021 | 1410.98 | 978.94 | 1309.36 | 2.79% |
| 2022 | 2343.32 | 2017.93 | 2286.90 | 11.73% |
| 2023 | 0.00 | 0.00 | 0.00 | 0.00% |
| 2024 | 797.88 | 581.40 | 761.95 | 12.58% |

2024 사후 최적 convex blend도 802.45점으로 v7보다 약 4.58점만 높다. 이전의
2024 outer-selected DL을 사용한 831점대 진단은 nested 검증에서 재현되지 않았다.

## bounded residual 구조

```text
p_final = sigmoid(logit(p_v7) + cap * tanh(ResidualMLP(meta, context)))
```

- final layer zero initialization
- Brier 직접 최적화
- correction L2와 평균 correction 패널티
- cap 후보는 사전에 정한 0.03 / 0.05 / 0.08
- 입력: v7·두 DL 확률, v7 seed 표준편차, 5-class·보조 head 확률,
  현재 행의 공식 context/as-of 피처
- 제외: 선수 ID 대형 embedding, `_game_cluster`, test 행 간 집계

## 2022 개발

선택 결과:

- cap: 0.05
- seed별 epoch: 42→3, 2026→3, 3407→2
- v7 Brier: 0.2433249059
- residual ensemble Brier: 0.2432928632
- ΔBrier: -0.0000320427
- 점수: 2343.32 → 2356.18
- 세 seed 모두 개선
- cap hit rate: 0

개발 게이트는 PASS했고, 이 설정으로 2021+2022 fixed-epoch bundle을 만들었다.

## 2023 체제 변화 stress

설정을 바꾸지 않고 동결 bundle을 적용했다.

| Slice | v7 Brier | residual Brier | ΔBrier |
|---|---:|---:|---:|
| 전체 | 0.25151922 | 0.25187244 | +0.00035322 |
| F | 0.27935505 | 0.28250358 | +0.00314853 |
| R | 0.24826688 | 0.24829349 | +0.00002662 |

사전에 정한 전체 악화 한도 +0.0005 안이라 형식적 stress gate는 통과했지만,
F 악화는 명백한 경고였다. 이 결과를 보고 모델·cap·게이트는 변경하지 않았다.

## 2024 최종 확인

| Slice | v7 Brier | residual Brier | ΔBrier | ΔScore |
|---|---:|---:|---:|---:|
| 전체 | 0.24781378 | 0.24790719 | +0.00009341 | -37.39 |
| F | 0.24729655 | 0.24739644 | +0.00009990 | -40.23 |
| R | 0.24788323 | 0.24797577 | +0.00009254 | -37.03 |

- v7 score: 797.88
- residual score: 760.48
- 세 seed 모두 v7보다 악화
- 평균 logit correction: +0.02259
- cap hit rate: 0

최종 확인 게이트는 FAIL이다. 2021에서 배운 양의 보정이 2022에서는 유효했지만,
2023·2024로 안정적으로 이동하지 않았다.

## 판정

현재 두 폴더를 결합하는 `v7 frozen base + DL residual`은 구현 가능하지만 성능상
채택할 수 없다. 단순 평균보다 복잡한 residual을 사용해도 체제 이동 문제가
해결되지 않았으며, nested 기준에서 DL의 추가 oracle 여지도 매우 작았다.

다음 연구는 이 residual의 cap·gate·연도 가중치를 재탐색하는 방식이 아니라,
v7이 사용하지 않는 독립적이고 연도 간 안정적인 새 신호를 먼저 찾아야 한다.
