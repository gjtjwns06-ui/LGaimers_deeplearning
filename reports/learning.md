# 딥러닝 학습 단계 계획 및 실행 계약

이 문서는 **학습 단계**만 다룬다. 각 development fold 내부에서 epoch 수를 정한 뒤, 새 모델을 outer train 전체에 고정 epoch로 다시 학습해 checkpoint를 만드는 것이 이 단계의 끝이다. Outer validation 예측·성능 검증, 2024 최근성 앵커, hyperparameter tuning, 앙상블·보정, 추론·제출은 수행하지 않는다.

## 1. 단계 경계

- 실행 fold는 `rolling_origin_2022`, `rolling_origin_2023`뿐이다.
- `rolling_origin_2024`는 다음 검증 단계의 명시적 앵커로 남겨 둔다.
- `recent_only_2023_to_2024` 진단 split은 실행하지 않는다.
- TrackMan과 구형 `FeaturePreprocessor`는 V1 학습 경로에 넣지 않는다.
- 모델 구조는 앞 단계에서 고정한 `context_gated_residual_v1`을 그대로 사용한다.
- 이 단계에서는 모델·optimizer·epoch 절차 사이의 공정한 비교 자체를 하지 않는다. 고정 절차 하나만 실행한다.

학습 runner에는 outer validation frame을 인자로 넘기는 경로가 없다. Outer validation은 전처리 `transform`, prediction, metric 계산, early stopping, checkpoint 선택에 모두 사용되지 않는다.

## 2. fold별 학습 흐름

```text
development outer fold
        │
        ├─ inner train만으로 전처리 fit
        │        │
        │        ├─ inner train 학습
        │        └─ inner stop Brier로 epoch 수만 선택
        │
        ├─ inner model·optimizer·가중치 폐기
        │
        └─ 새 전처리 + 새 모델 + 새 optimizer
                 │
                 └─ outer train 전체에서 선택된 epoch만큼 고정 학습
                            │
                            └─ checkpoint 저장
```

| Development fold | Inner epoch 선택 | Outer 고정 epoch refit | 이번 단계에서 접근하지 않는 영역 |
| --- | --- | --- | --- |
| `rolling_origin_2022` | 2019–2020 → 2021 stop | 2019–2021 전체 | 2022 outer validation |
| `rolling_origin_2023` | 2019–2021 → 2022 stop | 2019–2022 전체 | 2023 outer validation |

Inner stop에서는 Brier 하나만 epoch 선택 기준으로 사용한다. 여기서 선택된 것은 가중치가 아니라 **epoch 수와 이미 고정된 학습 절차**다. Outer refit은 inner checkpoint에서 이어서 학습하지 않고 같은 initialization seed로 새 모델을 생성한다.

## 3. 고정 학습 절차

사용할 기본값은 데이터나 성능을 본 뒤 바꾸지 않고 다음과 같이 고정한다.

| 항목 | 값 |
| --- | ---: |
| 최대 inner selection epoch | 8 |
| Early-stopping patience | 3 |
| Scheduler horizon | 8 epoch |
| Train batch size | 4,096 |
| Inner-stop eval batch size | 16,384 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.00001 |
| Loss | BCE-with-logits + 0.5 × Brier |
| Gradient clipping | 5.0 |
| Sample weighting | uniform |
| DataLoader workers | 0 |
| AMP | CPU 기본값에서 끔 |
| CPU threads | 8 |

Cosine scheduler의 horizon은 inner에서 선택된 epoch 수가 8보다 작더라도 8로 유지한다. 그래야 inner selection과 outer refit의 첫 `selected_epoch`개 최적화 경로가 같다.

## 4. 결정론과 seed 분리

기본 seed는 2026이고 fold 간 stride는 100이다. 초기화, dropout·학습, DataLoader shuffle seed를 분리한다.

| Fold | Initialization | Training/dropout | Loader shuffle |
| --- | ---: | ---: | ---: |
| `rolling_origin_2022` | 2026 | 2027 | 2028 |
| `rolling_origin_2023` | 2126 | 2127 | 2128 |

동일 실행 환경에서 동일 입력·config·seed의 재실행은 동일 selection record와 model state를 만들어야 한다. 다른 seed를 시도하거나 seed별 결과를 비교하는 일은 tuning·ensemble 단계로 미룬다.

## 5. checkpoint 계약

각 fold는 다음 구조로 저장한다.

```text
artifacts/learning/
├── rolling_origin_2022/
│   ├── selection_record.json
│   └── checkpoint/
│       ├── manifest.json
│       ├── preprocessor.json
│       ├── model_config.json
│       ├── model_state.npz
│       └── training_record.json
├── rolling_origin_2023/
│   └── ...
└── manifest.json
```

- `selection_record.json`에는 inner fit/stop row ID hash, 선택 epoch, 고정 절차, seed, epoch history를 기록한다. Inner model state와 prediction 배열은 저장하지 않는다.
- checkpoint에는 outer-train에서 fit한 전처리 상태, architecture config, 새로 학습한 model state, train loss history와 provenance를 저장한다.
- 전처리 fingerprint로 vocabulary·imputation·scaler와 모델 feature layout의 일치를 검증한다.
- 각 파일의 SHA-256과 tensor key·shape·dtype를 `manifest.json`에 기록한다.
- checkpoint 저장은 새 디렉터리에 원자적으로 수행하고 기존 checkpoint를 덮어쓰지 않는다.
- `--resume`은 selection record를 역직렬화하고 완성 checkpoint의 hash·구조·fold·seed·학습 절차·outer-validation 비접근 표시를 검증한 뒤 해당 refit을 건너뛴다.

## 6. 실행 방법

PowerShell 기준 기본 실행은 다음과 같다.

```powershell
python scripts/run_learning.py `
  --input data/train.csv `
  --output-dir artifacts/learning
```

중단 뒤에는 기존 산출물을 덮어쓰지 않고 다음처럼 검증·재개한다.

```powershell
python scripts/run_learning.py `
  --input data/train.csv `
  --output-dir artifacts/learning `
  --resume
```

`--folds`는 2022·2023의 일부만 허용하고 입력 순서와 관계없이 시간순으로 실행한다. 2024나 다른 연도를 넘기면 즉시 실패한다. 기본 device는 현재 환경에 맞춘 CPU이며 CUDA를 명시적으로 선택할 수 있지만, 절차 변경은 별도의 판단 없이 이 실행 안에서 하지 않는다.

## 7. 실행 결과 기록 위치

실행 완료 후 선택 epoch와 checkpoint hash는 `artifacts/learning/manifest.json`에 기록한다. 학습 loss는 각 checkpoint의 `training_record.json`, inner stop Brier 이력은 각 `selection_record.json`에 남긴다. 이 값은 optimizer가 정상 실행됐는지 재현하는 학습 기록이며 outer validation 성능으로 해석하지 않는다.

이번 단계에서 생성하는 prediction 파일과 outer validation metric은 **0개**다. 다음 검증 단계에서 checkpoint를 명시적으로 불러오기 전까지 outer validation 데이터는 계속 격리한다.

## 8. 실제 실행 결과 (2026-08-23)

고정한 기본 절차를 `data/train.csv` 1,475,092행에 변경 없이 실행했다. 입력 파일 SHA-256은 `d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff`다.

| Fold | Inner fit → stop | Inner 종료 / 선택 epoch | 선택 시점 inner Brier | Fresh outer-train 행 | Outer 고정 학습 loss |
| --- | --- | ---: | ---: | ---: | --- |
| `rolling_origin_2022` | 481,500행 → 247,088행 | 4 / **1** | 0.24605816 | 728,588 | 0.80614886 |
| `rolling_origin_2023` | 728,588행 → 247,472행 | 6 / **3** | 0.24377029 | 976,060 | 0.80543809 → 0.80269488 → 0.80119972 |

Inner Brier는 epoch 수를 고르기 위한 train-contained stop 값이다. Outer validation 성능으로 해석하거나 fold 간 모델 우열을 판단하는 데 사용하지 않았다.

저장 후 strict loader와 독립 `--resume` 실행이 모두 통과했다.

| Fold | Checkpoint manifest SHA-256 | Outer preprocessor fingerprint |
| --- | --- | --- |
| `rolling_origin_2022` | `d2cb38c07941c44a86b2f62cc0fb8ea79c8d692690e45ed3a1fa739f05a62033` | `3388d4b19005a5fa447a92723ff9441511617ad990be097c1f4a5a947abea85d` |
| `rolling_origin_2023` | `3bf716acc7a85abd44c043ed6225bb83f553a6dd6d856253cf17b4f6f2f8f65f` | `ad18b6e0858e8b7aee6ab7450c9281d0bf044585d79acfe56f07ffa57f7b0cde` |

최종 산출물 감사 결과는 총 13개 파일이며 prediction/OOF/submission/outer-metric 파일과 CSV·Parquet·Feather 출력은 모두 0개다. Manifest의 outer-validation transform/predict/metric counter는 두 fold 모두 0이고, 2024 anchor·recent-only diagnostic·tuning·보정·앙상블·추론·제출 실행 표시는 모두 `false`다.
