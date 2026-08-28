# F 체제 대응 딥러닝 후보 튜닝 결과

## 실행 경계

- 개발 입력: 2019--2023 전용 snapshot v2, 1,221,585행
- 외부 개발 평가: rolling-origin 2022, 2023
- 2024 recency confirmation 접근: 없음
- calibration, ensemble, test inference, submission: 수행하지 않음
- 불변 계획 SHA-256: `fc17e1c504984984d6096553746b439767ccdf0a2072002dfd2f00e7256af60c`
- 선택 SHA-256: `e61f432a3952512210c33bb40d901950d7e90d8dc04352e5027b98fc61e27637`

## 사전 등록 후보

- C1 `no_bias`: `game_type`의 직접 categorical bias만 제거
- C2 `neutral`: `game_type`을 직접 bias와 deep context 양쪽에서 제거
- C3 `bounded_history_residual`: C2 기반에 과거 투수 이력만 사용하는 `±0.5` bounded logit residual 추가

현재 투구의 reverse 여부, 2024를 나타내는 feature, target 파생 feature, 최근 시즌/F 가중치는 어느 후보에도 사용하지 않았다.

## 결과

괄호는 기존 V1 기준선 대비 Brier 차이다. 음수일수록 개선이다.

| 후보 | epoch 2022/2023 | 2022 Brier | 2023 Brier | 2023 F Brier | 2023 F 절대 bias | 2023 R Brier | hard gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| V1 baseline | 1 / 3 | 0.24432476 | 0.25500501 | 0.30336798 | 0.23298230 | 0.24935427 | fallback |
| C1 no_bias | 1 / 3 | 0.24480400 (+0.00047924) | 0.25452317 (-0.00048183) | 0.30038846 (-0.00297953) | 0.22571337 | 0.24916427 (-0.00019000) | PASS |
| **C2 neutral** | **1 / 1** | **0.24490141 (+0.00057665)** | **0.24987084 (-0.00513417)** | **0.26065309 (-0.04271489)** | **0.09484402** | **0.24861104 (-0.00074323)** | **PASS / selected** |
| C3 bounded history | 1 / 3 | 0.24478037 (+0.00045561) | 0.25208391 (-0.00292110) | 0.27830416 (-0.02506382) | 0.16589112 | 0.24902033 (-0.00033394) | PASS |

## 결정

세 후보 모두 다음 사전 hard gate를 통과했다.

1. 2022 전체 Brier가 V1 + 0.0025 이내
2. 2023 R Brier가 V1 + 0.0005 이내
3. 2023 F Brier가 V1보다 엄격히 개선
4. 2023 F 절대 bias가 V1보다 엄격히 개선

통과 후보에 고정된 lexicographic rule을 적용했다.

1. `max(Brier_2022, Brier_2023)`
2. `Brier_2023`
3. `abs_bias_2023_F`
4. `Brier_2023_F`
5. parameter count
6. candidate ID

최종 순위는 C2 → C3 → C1이며, `c2_neutral`을 잠갔다. C2는 V1 대비 2023 전체 Brier를 0.00513417, F Brier를 0.04271489 개선했고, F 절대 편향을 0.23298230에서 0.09484402로 약 59.3% 줄였다. 2022 Brier 손실은 0.00057665로 사전 허용 범위 안이다.

결과 디렉터리를 `--resume`으로 다시 열어 checkpoint, OOF row ID, metric 재계산, hard gate, ranking, candidate/selection hash를 검증했으며 동일 선택이 재현되었다.
