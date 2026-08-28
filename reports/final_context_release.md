# 최종 C2 딥러닝 릴리스 기록

## 결론

- 최종 후보: `c2_neutral` 단일 Context-Gated Residual 신경망
- 최종 학습: 2019~2024 전체 `1,475,092`행, fresh initialization, 고정 `1 epoch`
- 금지한 처리: warm start, validation 재사용, early stopping, 표본 가중치, TrackMan, calibration, ensemble
- 제출 후보: `submit_final_context.zip`

## 선택 및 2024 확인 근거

개발 구간의 C2 결과는 다음과 같다.

| 구간 | Brier | 추가 지표 |
| --- | ---: | ---: |
| 2022 전체 | 0.2449014050 | - |
| 2023 전체 | 0.2498708403 | - |
| 2023 F | 0.2606530925 | 절대 편향 0.0948440160 |
| 2023 R | 0.2486110414 | - |

봉인된 2024 원샷 확인은 `253,507`행에서 수행했고 모든 승격 게이트를 통과했다.

| 구간 | Brier | 실제 평균 | 예측 평균 | 절대 편향 |
| --- | ---: | ---: | ---: | ---: |
| 전체 | 0.2487437897 | 0.4861049202 | 0.4995082768 | 0.0134033566 |
| F | 0.2514938464 | 0.4592802399 | 0.5070759581 | 0.0477957182 |
| R | 0.2483745266 | 0.4897067970 | 0.4984921284 | 0.0087853314 |

전체 상수 기준선 Brier는 `0.2498069268`이었다. 2024 확인 결과는 전체 기준선 개선, 전체/게임 유형별 Brier 상한, F 절대 편향 `0.05` 상한을 모두 만족했다.

## 최종 학습 산출물

- 학습 환경: Python `3.11.15`, PyTorch `2.7.1+cpu`, NumPy `1.26.4`, pandas `2.0.3`
- 모델 파라미터: `349,858`
- epoch 1 학습 손실: `0.8094441426`
- artifact embedded manifest SHA-256: `765ca7be9c88bb551b5e2f7973d9c8b7f6bec7b9aace045e9b1f0a68b8f7899e`
- manifest 파일 SHA-256: `806e5de1791bd028e5776cddac60995316e7bd3379d51efdce14c08aa7067967`
- 모델 상태 SHA-256: `85595dc8a0468992cf0ec472d28e82bda3a04535cfa2de71916d21214c27e29e`
- NPZ 파일 SHA-256: `97f0bcc3aefcb4ea5d273cb7c53299fbeea6eee5451c54218a1534e6726a7ae6`

Windows/PyTorch 2.13에서 먼저 만든 산출물은 공식 서버의 초기화 해시와 호환되지 않아 `artifacts/final_context_c2_incompatible_torch213`에 격리했다. 제출에는 사용하지 않는다.

## 검증 결과

- Windows 개발 환경 전체 테스트: `97 passed`
- 공식 호환 환경 전체 테스트: `97 passed`
- 공식 호환 환경 strict artifact resume/load: 통과
- target-free, 입력 순서 보존, singleton/배치 동일성, 미등록 범주 처리: 통과
- artifact exact inventory, source/anchor/code hash, strict NPZ, fresh single-model 의미 계약: 독립 감사 통과

## 제출 ZIP

- 경로: `submit_final_context.zip`
- SHA-256: `6d93702ab46cf6c5d96bf2d43b7283cc982227799e80ee5a69e45ee1f1ae1692`
- 압축 크기: `1,414,574` bytes
- 압축 해제 크기: `1,819,795` bytes
- 루트: `script.py`, `requirements.txt`, `model/`만 존재
- 내부 구성: 30 entries, 24 files, 중복/unsafe path/symlink 없음
- 코드 봉인: artifact가 고정한 15개 파일만 `model/runtime`에 포함하고 서버 실행 전 SHA-256 대조
- 배포 봉인: entry, requirements, runtime, bundle 전체 23개 파일을 deployment manifest로 대조

공식 평가 규모 `245,789`행을 6 CPU affinity에서 격리 실행한 결과:

- 추론 및 출력 검증: `10.62`초
- 자식 프로세스 peak RSS: `1,056.34 MiB`
- requirements 설치 단계: `0.61`초

이는 공식 제한인 설치/추론 각 10분, ZIP 10GB, 압축 해제 32GB, RAM 28GB보다 충분히 작다.

## 주의

비공개 평가 데이터의 정답은 제공되지 않으므로 최종 ZIP의 실제 리더보드 Brier/Score는 로컬에서 계산할 수 없다. 위 성능 수치는 봉인된 2024 확인 데이터 결과이며, 제출 후 점수와 구분해야 한다.
