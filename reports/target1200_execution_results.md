# 1200점 목표 딥러닝 실행 결과

실행일: 2026-08-24  
검증: `train < 2024`, `validation = 2024`, 253,507행 forward validation

## 결론

현재 최고 고정 조합은 `rate-basis 5-class v2`와 `prior-season target-TE` 모델의
동일가중 평균이다. 2024 Brier는 **0.24798755**, 환산 점수는 **728.31점**이다.
F slice gate는 통과하지만 전체 800점 gate에는 Brier **0.00017908**이 부족하다.

따라서 현재 후보는 제출 모델로 승격하지 않았다. 2024 라벨로 구한 oracle 가중치,
사후 보정값, 실패한 교차시점 보정도 제출 경로에 포함하지 않았다.

| Gate | Brier 상한 | 현재 차이 | 판정 |
|---|---:|---:|---|
| 800 | 0.24780847 | +0.00017908 | FAIL |
| 1,000 | 0.24730886 | +0.00067869 | FAIL |
| 1,200 | 0.24680924 | +0.00117831 | FAIL |

## 현재 최고 고정 평균

| Slice | 행 수 | 실제 평균 | 예측 평균 | Brier | 환산 점수 |
|---|---:|---:|---:|---:|---:|
| 전체 | 253,507 | 0.486105 | 0.489615 | **0.24798755** | **728.31** |
| F | 30,010 | 0.459280 | 0.463003 | 0.24741589 | 372.88 |
| R | 223,497 | 0.489707 | 0.493189 | 0.24806431 | 732.20 |

동일가중 평균은 두 모델에 정확히 0.5씩 적용했다. 2024에서 계산한 최적 가중치
`0.543/0.457`은 진단값일 뿐 채택하지 않았다. 다만 두 구성원의 선택 자체는 같은
2024 anchor 결과를 비교한 탐색 결과이므로, 728.31점도 최종 unbiased 추정치가 아니라
낙관 가능성이 있는 로컬 진단값이다. 이 진단값조차 Gate1에 못 미친다.

## 주요 전체 검증 결과

| 후보 | Brier | 환산 점수 | 판정 |
|---|---:|---:|---|
| base Cross | 0.24826492 | 617.28 | 폐기 |
| base multi-task | 0.24823806 | 628.03 | 폐기 |
| structured multi-task | 0.24823723 | 628.36 | 폐기 |
| prior-season target TE | 0.24818669 | 648.60 | 평균 구성원 |
| strict prior TrackMan + TE | 0.24820109 | 642.83 | TrackMan 증분 실패 |
| compact rate-basis multi-task | 0.24818488 | 649.32 | 5-class 기반 입력 |
| rate + target-TE 단일 결합 | 0.24834564 | 584.97 | 고차원 결합 실패 |
| rate-basis 5-class v2, seed 42 | 0.24812814 | 672.03 | 평균 구성원 |
| 5-class 저학습률 미세학습 | 0.24811669 | 676.62 | 다양성 감소 |
| 5-class Brier×4 미세학습 | 0.24811540 | 677.13 | 다양성 감소 |
| rate-basis 5-class v2, seed 2026 | 0.24824025 | 627.16 | seed 불안정 |
| periodic/piecewise numeric DL | 0.24825947 | 619.46 | 폐기 |
| targeted prior + F/R split head | 0.24837518 | 573.14 | 폐기 |
| target-TE 5-class | 0.24828341 | 609.88 | 폐기 |

## 구조 감사와 안전장치

- 4-class 초안은 reverse와 middle이 동시에 양성인 41,826행을 발견한 즉시 학습 전
  중단했다. 최종 v2는 `success`, `reverse_only`, `middle_only`,
  `reverse_and_middle`, `other_failure`의 정확한 5-class다.
- 2019--2023 fold-train에서 1,220,874행은 5-class로 복원했고, 말단 711행은
  임의 class를 만들지 않고 binary fallback loss만 사용했다.
- strict current-pitch TrackMan 보조라벨은 validation-year 이상 TrackMan을 정렬 전에
  차단한다. 2019--2023에서 659,701행(54.00%)만 strict match됐다.
- 2022→2023에서만 계수를 학습한 residual correction은 2024에서
  `0.24798755 → 0.24824622`로 악화해 폐기했다.
- 전체 테스트는 **192 passed**다.

## 다음 승격 조건

추가 실험은 현재 동일가중 평균보다 Brier를 최소 0.00017908 낮추고, 두 seed 이상에서
방향이 일치해야 한다. 이 조건 전에는 calibration, oracle ensemble, 최종 재학습,
추론 ZIP 생성을 진행하지 않는다. 다음 고위험 후보는 strict matched TrackMan의 현재
구종/물리량을 train-only auxiliary 또는 OOF teacher로 쓰는 방법이지만, 이전 진단의
current-TrackMan 단독 신호가 약했으므로 전체 2024 Brier 증분 0.00010 gate를 적용한다.
