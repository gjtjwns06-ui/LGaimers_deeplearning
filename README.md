# LG Aimers 투구 제구 성공 확률 예측

규칙을 위반하지 않는 시간순 검증, 범주 임베딩 Residual MLP, OOF 확률
보정, 독립 실행형 제출 ZIP을 포함한 재현 가능한 파이프라인입니다.

## 핵심 원칙

- 검증은 `season < validation_year`로만 학습하는 expanding split을 사용합니다.
- 평가 행을 정렬·집계·차분·rolling하지 않습니다.
- 공식 `asof_*` 값은 현재 행 입력으로만 사용합니다.
- `row_id`는 모델 입력에 포함하지 않습니다.
- 범주 사전, 수치 통계, smoothing prior는 학습 fold에서만 계산합니다.
- 테스트 평균이나 분포를 이용해 확률을 보정하지 않습니다.
- 제출 추론은 다른 행의 존재, 순서, batch 크기와 무관합니다.

## 검증 결과

모든 값은 전체 시즌 홀드아웃 Brier이며 낮을수록 좋습니다.

| Validation | RF | Residual MLP |
| --- | ---: | ---: |
| 2022 | 0.243987 | **0.243453** |
| 2023 | **0.251651** | 0.254270 |
| 2024 | 0.248769 | **0.248413** |

2023년은 `game_type=F` 성공률이 2022년 70.87%에서 47.29%로 단절된
시점이라 과거 시즌만으로 외삽한 두 모델이 모두 약했습니다. 실제 2025와 가장
가까운 2024 홀드아웃을 주 선택 지표로 사용했습니다.

2024에서 세 seed의 모든 동일가중 조합을 비교한 결과 seed 42+3407이
선택됐습니다.

- 2-model raw Brier: `0.24837308`
- 2024 OOF affine-logit 보정 후: `0.24832892`
- 2024 기준 Brier Skill: `0.59166%` (점수식 환산 `591.66`)

이는 로컬 시간 홀드아웃 결과이며 실제 리더보드 점수를 보장하지 않습니다.

## 빠른 검증

PowerShell 기준입니다.

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
python scripts/package_submission.py --output submit.zip --self-check
```

현재 테스트는 스키마, OOV/결측 처리, 저장 왕복, 확률 보정, Trackman cutoff,
단독·batch·shuffle 행 독립성을 검사합니다.

## 학습 순서

### 1. RF 시간 기준선

```powershell
python scripts/train_baseline.py `
  --folds 2022 2023 2024 `
  --output-dir artifacts/baseline
```

### 2. 딥러닝 시간 CV

```powershell
python scripts/train_deep.py `
  --folds 2022 2023 2024 `
  --epochs 8 --patience 3 `
  --hidden-dim 128 --num-blocks 3 `
  --batch-size 8192 --eval-batch-size 32768 `
  --output-dir artifacts/deep_cv
```

GPU가 있으면 자동으로 CUDA를 사용합니다. CPU 스모크에는
`--max-train-rows`, `--max-val-rows`, `--device cpu`를 사용합니다.

### 3. seed 조합과 보정

각 seed의 같은 validation 행 예측을 만든 뒤 다음처럼 조합을 선택합니다.

```powershell
python scripts/evaluate_ensemble.py `
  --prediction 42=artifacts/deep_cv_seed42/oof_predictions.csv `
  --prediction 2026=artifacts/deep_cv_seed2026/oof_predictions.csv `
  --prediction 3407=artifacts/deep_cv_seed3407/oof_predictions.csv `
  --output-dir artifacts/ensemble_selection

python scripts/calibrate_oof.py `
  --input artifacts/ensemble_selection/selected_oof.csv `
  --min-season 2024 `
  --output artifacts/ensemble_selection/calibrator.json
```

보정은 테스트 데이터가 아니라 시간 OOF 예측과 학습 정답만 사용합니다.

### 4. 전체 데이터 최종 학습

```powershell
python scripts/train_final.py `
  --seeds 42 2026 3407 `
  --epochs 2 --hidden-dim 128 --num-blocks 3 `
  --batch-size 8192 `
  --output-dir artifacts/final_bundle

python scripts/select_bundle_models.py `
  --input-bundle artifacts/final_bundle `
  --output-bundle artifacts/final_deploy_bundle `
  --model-indices 0 2 `
  --calibrator artifacts/ensemble_selection/calibrator.json
```

배포 bundle은 PyTorch 버전 간 pickle 호환 문제를 줄이기 위해 가중치를
NPZ로 저장하고 SHA-256으로 검증합니다.

## 선택적 Trackman 실험

메인 선수 ID와 Trackman 선수 ID는 연결하지 않습니다. 현재 투구 물리값도
붙이지 않으며, 각 대상 시즌보다 이전 시즌의 공통 경기 상황만 집계합니다.

```powershell
python scripts/build_trackman_context.py `
  --output artifacts/trackman_context.csv `
  --target-seasons 2019 2020 2021 2022 2023 2024 2025

python scripts/train_deep.py `
  --folds 2024 `
  --trackman-lookup artifacts/trackman_context.csv `
  --output-dir artifacts/deep_trackman
```

Trackman 경로는 구현 및 스모크 검증까지만 완료됐고 전체 CV 개선이 확인되지
않았으므로 현재 최종 제출에는 포함하지 않았습니다.

## 제출 ZIP

```powershell
python scripts/package_submission.py `
  --output submit.zip `
  --smoke-data data/test.csv
```

ZIP 최상위는 다음 세 항목뿐입니다.

```text
submit.zip
├── script.py
├── requirements.txt
└── model/
```

`script.py`는 `./data/test.csv`를 우선 사용하고 공식 페이지의 상충 문구에
대비해 `./open/test.csv`도 fallback으로 지원합니다. 출력은 테스트의
`row_id`에서 직접 생성하며 누락·중복 ID, NaN/Inf, 범위 밖 확률이면 즉시
실패합니다. 현재 최종 모델의 CPU 합성 벤치마크는 245,789행에 약 9.2초입니다.

