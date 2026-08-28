# 딥러닝 모델 설계 및 구현 결과

이 문서는 **모델 설계 준비와 구조 구현**까지만 다룬다. optimizer 실행, 가중치 학습, 모델 예측 성능 검증, epoch·hyperparameter 선택, 튜닝, 앙상블·보정, 추론·제출은 수행하지 않았다.

## 1. 설계 경계

- canonical 입력은 `DeepTabularPreprocessor`의 범주형 17개·수치형 60개로 확정한다.
- 모델 설계에는 `rolling_origin_2022`, `rolling_origin_2023`만 development fold로 노출한다.
- 실제 입력·파라미터 감사의 참조 영역은 2019–2022 train → 2023 validation이다.
- `rolling_origin_2024`의 모델 예측·성능은 이번 실행에서 계산하지 않았고 architecture config에도 2024행을 사용하지 않았다. 다만 앞선 EDA·검증 설계에서 2024 정답 분포를 이미 관찰했으므로 blind/untouched holdout이라고 부르지 않는다.
- 전처리 vocabulary·중앙값·평균·표준편차는 참조 fold의 train 976,060행에서만 fit했다.
- 모델은 행마다 독립적인 logit 하나를 반환한다. sigmoid, loss, metric은 모델 밖의 이후 단계 책임이다.

기존 `FeaturePreprocessor`와 과거 학습·배포 bundle은 호환성 확인용 legacy 경로로 남겨 둔다. 새 모델 경로에서 구형 전처리와 신형 전처리를 섞지 않는다.

## 2. 데이터가 요구하는 구조

### 미래 시즌과 선수 OOV

시간 검증에서는 다음 시즌의 `season` 범주가 항상 100% OOV다. 2023 validation 기준 투수 ID OOV는 13.80%, 타자 ID OOV는 9.40%였다. 2024에서는 이전 시즌에 두 선수가 모두 등장한 행도 72.78%에 그친다.

따라서 season·선수 embedding을 주 예측 경로로 쓰지 않는다. missing/OOV index 0·1의 embedding은 forward에서 0으로 만들고, 연속형 `season_numeric`, 공식 `asof_*`, 경기 문맥으로 후퇴한다.

### 이력과 결측

ID가 OOV여도 공식 `asof_*` 이력은 존재할 수 있다. 그래서 선수 ID와 선수 이력을 같은 입력으로 묶어 통째로 제거하지 않는다. ID는 history tower 위에 더하는 작은 residual correction으로만 사용한다. 이력 횟수·cold-start·결측 mask는 gate 입력에 포함된다.

결측 mask는 기존 전처리 계약을 변경하지 않고 명시적 numeric feature로 유지하되, 해당 투수·타자 tower에 이름으로 배치한다. 별도의 unscaled binary branch는 primary V1의 성능을 확인하기 전에는 추가하지 않는다.

### 시즌×경기유형 단절

`game_type=F`는 expanding train→validation에서 2023 -20.32%p, 2024 -17.70%p의 타깃률 변화를 보였다. 대형 F 전용 expert는 과거 정의를 암기할 위험이 있어 사용하지 않는다. `game_type` 문맥 embedding, `season_numeric`, context/entity 상호작용을 공유 trunk에서 저용량으로 학습하게 한다.

## 3. 후보 비교와 선택

| 후보 | 장점 | 결정 |
| --- | --- | --- |
| 기존 Residual MLP | 단순하고 빠른 row-local 기준선 | baseline으로 유지 |
| Context-Gated Residual | 명시적 선수 fallback, 문맥·이력 분리, 작은 자원 | **primary V1로 구현** |
| Low-rank DCNv2 | 저비용 명시적 cross | 향후 tuning 후보로 보류 |
| FT/TabTransformer | 유연한 feature interaction | 활성화 비용·과확신 위험 때문에 V1에서 제외 |

TrackMan은 메인 선수 ID와 교집합이 없고 2022년 수집 체제 단절도 있어 core V1에서 제외한다. 선수×타자 pair vocabulary와 행 순서 기반 LSTM·GRU·Transformer도 사용하지 않는다.

## 4. 구현한 architecture

```text
이름 기반 FeatureLayout
        │
        ├─ context 범주·수치 ────────────────> Context tower ────────┐
        │                                                            │
        ├─ 투수 손/팀 + asof 이력 ───────────> Pitcher base tower ───┼─> 행 내부 상호작용
        │        └─ known ID × reliability gate × ID residual ───────┤   [p,b,c,p*b,p*c,b*c,|p-b|]
        │                                                            │             │
        └─ 타자 손/팀 + asof 이력 ───────────> Batter base tower ────┘             v
                                                                    gated-GELU residual trunk
전체 수치형 ───────────────────────────────────────────────> numeric wide logit      │
선수 ID를 제외한 known 범주 ───────────────────────────────> categorical bias logit │
                                                                                    v
                                                bias + wide + categorical + deep = logit
```

고정한 V1 기본값은 다음과 같다.

| 항목 | 값 |
| --- | ---: |
| Tower width | 64 |
| Residual hidden width | 128 |
| Gated-GELU residual blocks | 3 |
| FFN width | 192 |
| Dropout | 0.15 |
| Entity embedding dropout | 0.10 |
| Embedding dimension cap | 32 |
| Parameter budget | 1,000,000 미만 |

architecture의 초기 output bias는 타깃 통계와 분리해 항상 logit 0, 즉 확률 0.5로 고정한다. 이후 학습 단계에서 train prior 초기화를 채택하려면 `FitScopeAudit`로 해당 fold train만 사용했음을 강제하는 별도 API를 먼저 구현해야 한다. 이 모델 config를 통해 fold 타깃률을 직접 넘길 수는 없다.

## 5. 안전 계약

- 피처 그룹과 선수 위치는 숫자가 아닌 컬럼명으로 해석한다. 신형 전처리의 실제 투수·타자 위치 6·7을 확인했다.
- `row_id`, `control_success`는 layout에 들어가면 즉시 실패한다.
- categorical은 `int64`이고 각 index가 해당 cardinality 내부여야 한다.
- numerical은 `float32`이며 모두 유한해야 한다.
- 모든 missing/OOV categorical embedding은 0으로 후퇴한다.
- 선수 ID gate는 missing/OOV에서 정확히 0이다. 선수 history/context 경로는 계속 작동한다.
- 선수 ID에는 categorical scalar bias를 만들지 않아 entity dropout과 reliability gate를 우회하는 ID 경로가 없다.
- 모델 layout에는 vocabulary token→index와 수치 imputation·scaler 전체 상태의 SHA-256 fingerprint가 포함된다. 같은 크기라도 인코딩이 다르면 호환 검사가 실패한다.
- BatchNorm, batch 축 집계, 행 간 attention을 사용하지 않는다.
- 동일 seed의 모델 초기 state는 완전히 같고 다른 seed에서는 달라야 한다.
- eval mode에서 한 행 단독, batch, 행 순열 후 복원 결과가 허용 오차 내 같다.
- config는 JSON round-trip이 가능하고 state 복원 뒤 logit이 정확히 일치한다.

## 6. 실제 데이터 실행 결과

- 입력: 전체 1,475,092행의 공식 train schema
- 전처리 fit: 2019–2022의 976,060행만 사용
- 입력 audit: 2023의 245,525행, categorical `int64` 17열, numerical `float32` 60열, 전부 유한
- 실제 model parameters: **340,330개**
- FP32 weight 크기: **1.298 MiB**
- 초기화 forward smoke: 16행에서 finite logit `[16]`
- 단일 행 대 batch 최대 절대 차이: `9e-8` 미만
- 행 역순 후 복원 최대 절대 차이: `0.0`
- 모델 설계 신규 테스트 10개, 전체 회귀 테스트 40개 통과
- 모델 설계 산출물은 동일 입력 재실행 전후 SHA-256이 동일
- 저장한 학습 weight·prediction·metric: 없음

## 7. 구현 산출물

- `src/lg_control/model.py`: feature layout, config, 3-tower 모델, seeded factory, 입력·파라미터 감사 API
- `src/lg_control/validation.py`: development folds와 2024 anchor의 명시적 분리
- `tests/test_model_design.py`: layout, OOV fallback, seed, 행 독립성, config/state, 입력 오류 테스트
- `scripts/build_model_design.py`: development scope에서 실제 입력 규격과 초기화 architecture만 재현하는 실행 스크립트
- `artifacts/model_design/manifest.json`: 입력 hash, 선택 경계, architecture config, 파라미터·OOV·forward audit
- `artifacts/model_design/feature_layout.csv`: 피처별 위치·그룹·cardinality·embedding 크기·OOV율
- `artifacts/model_design/parameter_summary.csv`: module별 파라미터 수
- `artifacts/model_design/candidate_comparison.csv`: 후보와 보류 사유

## 8. 다음 단계로 넘기는 항목

다음 학습 단계에서는 기존 `train_deep.py`를 그대로 재사용하지 않는다. 그 스크립트는 구형 전처리와 outer validation early stopping 경로를 사용한다. 신규 runner는 반드시 `inner train → inner stop에서 epoch 절차 확정 → 초기화 → outer train 고정 epoch fit` 계약을 사용해야 한다.

이번 단계에서는 loss·optimizer 실행, epoch 선택, Brier 계산, 모델 간 성능 비교, hyperparameter 변경을 하지 않았다.
