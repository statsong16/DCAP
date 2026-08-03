# 02 — 다중오믹스 층위화

**상태:** 합성 데이터 검증 완료 (실제 데이터 미적용)

## 질문

microbiome · metabolome · proteome 세 블록을 합쳐 비지도로 표본의 **층위(stratum)** 를
나눌 수 있는가. 나눴다면 그 층위가 메타데이터(BMI·성별·연령·enterotype)를
설명하는가, 그리고 held-out 표본에서 메타데이터를 맞추는 데 쓸 수 있는가.

## 데이터

| 항목 | 값 |
|---|---|
| 출처 | `multi_omics_set.RData` — 공개/게재된 자료. 저장소에는 포함하지 않으므로 루트에 직접 두세요 |
| 구조 | 피험자 69명 × 방문 4회 = 표본 276개, 결측 없는 완전 설계 |
| microbiome | 1350 MSP × 276, 0 비율 0.839, 열 합이 표본마다 다름 (CV 0.12) |
| metabolome | 413 × 276, 원시 강도 3.7e3–1.0e11, NA 1.45% |
| proteome | 770 × 276, Olink NPX(log2) −3.10–13.30, NA 1.67% |
| 접근 | `python src/export_real.py --rdata ../../multi_omics_set.RData --out data` |
| 합성 대체 | `src/make_synthetic.py` — 반복측정·피험자 효과·잠재 층위 포함 |
| **아래 결과의 데이터** | **합성** (실제 자료로는 아직 돌리지 않았습니다 — CLAUDE.md §5) |

`type`(Case), `Geography`(Sweden), `Sequencer`(HiSeq)는 276개 표본에서 모두 같은
값이라 표적으로도 공변량으로도 쓸 수 없습니다.

## 방법

**분할이 이 실험의 핵심 설계 결정입니다.**
한 사람에게서 4회 측정이 나왔고, Age·Gender·BMI 는 사실상 피험자 상수입니다
(피험자 내 SD: Age 0.34년, BMI 0.30). 표본 단위로 나누면 같은 사람이 train 과 test
양쪽에 있고, 모델은 표현형이 아니라 **사람**을 외우면 됩니다. 모든 분할은
`metadata.ID` 기준 GroupKFold 이고, 전처리·PCA·KMeans 는 fold 마다 train 에서만
다시 fit 합니다. `--leaky-split` 으로 일부러 틀리게 나눈 대조도 같이 돌립니다.

**블록별 전처리** — 값의 성격이 달라 같은 변환을 쓸 수 없습니다.

| 블록 | 처리 |
|---|---|
| microbiome | 존재율 10% 필터 → TSS → pseudocount(train 최소 비영값의 절반) → CLR |
| metabolome | log10(x+1) → train 중앙값 대치 |
| proteome | NA 20% 초과 feature 제거 → train 중앙값 대치 (NPX 는 이미 log2) |

CLR 을 먼저 걸지 않으면 조성의 폐쇄성에서 오는 가짜 음의 상관이 그대로 들어옵니다.

**결합** — 블록별 PCA 10개씩 → 30차원. 그대로 이어붙이면 feature 가 1350개인
microbiome 이 거리를 독점하므로 블록마다 같은 지분을 줍니다. SNF·MOFA 같은 방법도
있지만 표본 276개에 그 복잡도를 얹을 근거가 아직 없습니다 (CLAUDE.md §2).

**층위화** — KMeans, k 는 실루엣으로 k=2..6 에서 선택. 고른 뒤 세 가지를 같이 봅니다:
피험자 부트스트랩 ARI(안정성), 피험자 일관성(한 사람의 4회 방문이 같은 군집인 비율),
메타데이터 연관(피험자 단위 n=69, BH 보정).

## 실행

```bash
export PYTHONPATH=src
bash run_all.sh            # 합성 데이터로 전체
```

| 항목 | 값 |
|---|---|
| 시드 | 0 (모든 스크립트 `--seed 0`) |
| 분할 | 피험자 단위 GroupKFold 5-fold |
| 임베딩 | 블록당 PCA 10개 → 30차원 |
| 실행 시간 | 71초 (CPU) |
| 환경 | Python 3.11.15, numpy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0, scipy 1.17.1, rdata 1.1.0 |

## 결과

아래는 전부 **합성 데이터** (표본 276, 피험자 69, 잠재 층위 3개, 시드 0) 결과입니다.

### 층위화 (전체 자료 적합 — held-out 추정치 아님)

| 항목 | 값 |
|---|---|
| 선택된 k (실루엣) | 2 (실루엣 0.080) |
| 군집 크기 | 87 / 189 |
| 부트스트랩 ARI | 0.754 ± 0.270 (피험자 리샘플 20회) |
| 피험자 일관성 | 0.986 |
| 정답 층위와의 ARI | 0.612 (합성 자료에서만 채점 가능) |
| 정답 층위로 설명되는 BMI 분산 | 0.206 (= 아래 BMI R² 의 상한) |
| PC1–10 설명 분산 | microbiome 7.0% · metabolome 13.8% · proteome 18.1% |

실루엣 곡선: k=2 0.080 · k=3 0.068 · k=4 0.067 · k=5 0.075 · k=6 0.066.

**심어 둔 층위는 3개인데 실루엣은 2개를 골랐습니다.** 정답과의 ARI 0.612 는
지배적인 분리는 잡았지만 세 번째 층위는 놓쳤다는 뜻입니다.

메타데이터 연관 (피험자 단위 n=69, BH 보정):

| 변수 | 검정 | 통계량 | p | q(BH) |
|---|---|---|---|---|
| Age | Kruskal-Wallis | 5.374 | 0.0204 | 0.102 |
| Gender | chi-square | 2.979 | 0.0843 | 0.211 |
| BMI | Kruskal-Wallis | 1.255 | 0.263 | 0.438 |
| enteroType | chi-square | 0.631 | 0.729 | 0.912 |
| subtype (표본 단위) | chi-square | 0.050 | 0.997 | 0.997 |

**보정 후 유의한 것은 없습니다.** 합성 자료에서 층위와 실제로 연결해 둔 변수는
BMI 하나뿐인데 그 BMI 는 못 잡았고, 아무 관계도 심지 않은 Age 가 보정 전 p=0.02 로
나왔습니다. 피험자 69명에 검정 5개면 이런 일이 일어납니다. 보정 전 p 만 보고
"층위가 연령과 연관된다"고 쓰면 그대로 틀린 문장이 됩니다.

### 메타데이터 예측 (피험자 단위 GroupKFold 5-fold, pooled OOF)

회귀 (R² 는 높을수록, MAE 는 낮을수록 좋음):

| 표적 | 특징 | 모델 | R² | MAE |
|---|---|---|---|---|
| BMI | trivial | – | −0.024 | 2.421 |
| BMI | strata_onehot | ridge | 0.014 | 2.380 |
| BMI | joint_embedding | ridge | **0.106** | 2.238 |
| BMI | joint_embedding | gbdt | −0.193 | 2.591 |
| BMI | microbiome_pcs | ridge | −0.020 | 2.427 |
| BMI | metabolome_pcs | ridge | 0.058 | 2.272 |
| BMI | proteome_pcs | ridge | **0.117** | 2.236 |
| Age | trivial | – | −0.037 | 3.088 |
| Age | strata_onehot | ridge | 0.018 | 2.975 |
| Age | joint_embedding | ridge | −0.086 | 3.129 |
| Age | proteome_pcs | ridge | 0.011 | 3.016 |

분류 (pooled OOF):

| 표적 | 특징 | 모델 | AUC | Brier | 보정기울기 |
|---|---|---|---|---|---|
| Gender=Female | trivial | – | 0.402 | 0.255 | −4.03 |
| Gender=Female | strata_onehot | logreg | 0.565 | 0.242 | 0.73 |
| Gender=Female | joint_embedding | logreg | 0.502 | 0.282 | 0.02 |
| Gender=Female | proteome_pcs | logreg | 0.565 | 0.249 | 0.52 |
| enteroType=ET-Firmicutes | trivial | – | 0.435 | 0.190 | −4.41 |
| enteroType=ET-Firmicutes | strata_onehot | logreg | 0.422 | 0.191 | −2.78 |
| enteroType=ET-Firmicutes | joint_embedding | logreg | 0.496 | 0.197 | 0.00 |
| enteroType=ET-Firmicutes | microbiome_pcs | gbdt | 0.557 | 0.255 | 0.07 |

전체 표(6개 특징 × 2개 모델 × 4개 표적)는 `runs/predict/results.csv` 에 있습니다.

`trivial` 은 특징 없이 train 의 평균/사전확률만 내놓습니다. fold 마다 그 값이 조금씩
달라서 pooled AUC 가 정확히 0.5 가 아니고 보정기울기도 의미가 없습니다. 이 행은
Brier 만 읽으세요.

### 누수 대조 (`--leaky-split`, 보고용 아님)

같은 파이프라인을 **표본 단위**로 나눴을 때. 유일한 차이는 분할입니다.

| 표적 | 특징 | 모델 | 피험자 단위 | 표본 단위 |
|---|---|---|---|---|
| Gender=Female (AUC) | proteome_pcs | gbdt | 0.455 | **0.982** |
| Gender=Female (AUC) | joint_embedding | gbdt | 0.485 | **0.963** |
| BMI (R²) | proteome_pcs | gbdt | −0.334 | **0.660** |
| BMI (R²) | joint_embedding | gbdt | −0.193 | **0.542** |
| Age (R²) | proteome_pcs | gbdt | −0.123 | **0.641** |

합성 자료에는 성별 신호도 연령 신호도 심지 않았습니다. 그런데 표본 단위로 나누면
성별 AUC 0.98 이 나옵니다. 전부 같은 사람의 다른 방문을 맞춘 것입니다.

## 해석

**합성 자료에서 층위는 잡히지만, 그 층위로 메타데이터를 맞추지는 못했습니다.**
BMI 에서 `joint_embedding` ridge 0.106, `proteome_pcs` ridge 0.117 로 trivial
(−0.024)보다 나은 것이 전부입니다. 정답 층위를 그대로 알려줬을 때의 상한이 0.206
(`stratify.py` 가 계산)이므로, 있는 신호의 절반 정도를 건진 셈입니다.
Gender·Age·enterotype 은 전부 우연 수준이고, 이는 합성 자료에 그 신호를 심지
않았으므로 **정확히 그래야 하는 결과**입니다.

**군집 원-핫(`strata_onehot`)이 30차원 임베딩보다 나은 칸이 여러 곳입니다.**
BMI gbdt 에서 0.014 vs −0.193. 군집 번호 2개는 사실상 강한 정규화이고, 피험자 69명
문제에서 30차원 gbdt 는 그냥 과적합합니다. 층위화의 값어치가 "새 정보를 만든다"보다
"차원을 줄인다" 쪽에 있을 수 있다는 신호입니다.

**gbdt 가 선형 모델에 거의 전부 졌습니다.** 회귀 10개 칸(특징 5 × 표적 2) 중 ridge 가
9칸에서 이겼고, gbdt 가 이긴 1칸은 두 값이 소수점 셋째 자리까지 같은
`strata_onehot` 입니다. 표본 276개(실질 피험자 69명)에 부스팅을 쓸 근거가 없다는
뜻이고, 실험 01 에서 부스팅이 이겼던 것과 반대 방향입니다 — 거기는 표본 360개에
신호가 강했습니다.

**보정기울기는 전부 1보다 한참 작습니다.** 가장 나은 칸도 0.73 입니다. AUC 만 보고
넘어가면 안 되는 이유입니다 (CLAUDE.md §9).

**누수 대조가 이 실험에서 제일 확실한 결과입니다.** 아무 신호도 없는 표적에서
분할 방식만 바꿔 AUC 0.485 → 0.963 이 나옵니다. 이 자료로 논문을 쓸 때 표본 단위
교차검증을 했다면 그 수치는 전부 사람 재인식입니다.

## 한계

- **위 수치는 전부 합성 자료입니다.** 실제 `multi_omics_set.RData` 로 돌린 결과는 아직
  이 저장소에 없습니다. `src/export_real.py --out data` 로 내보내면 `run_all.sh` 의
  나머지가 그대로 돌아가는 것까지는 확인했습니다.
- **§7 을 어디까지 적용했는지.** 원자료는 공개/게재된 자료이므로 §7 의 프라이버시
  근거는 약합니다. 그래도 결과 표는 합성 자료로만 채웠는데, 이건 프라이버시가 아니라
  "저장소가 자료 접근 없이도 클론 하나로 재현되어야 한다"는 쪽 이유입니다. 위 "데이터"
  표의 파일 규격(행렬 크기·결측률·값의 스케일)은 전처리 근거를 적으려면 필요해서
  남겼고, 코호트 구성(성별·enterotype 분포, BMI 요약값)은 연구 결과라 적지 않았습니다.
  실제 자료 수치를 README 에 싣기로 정하신다면 §5 에 따라 시드와 함께 실행한 결과만
  넣으면 됩니다.
- **k 를 전체 자료에서 골랐습니다.** `predict.py` 는 `stratify.py` 가 전체 자료 실루엣으로
  고른 k 를 받아 쓰므로, 이 한 개 하이퍼파라미터에는 약한 선택 누수가 있습니다.
  전처리·PCA·KMeans 적합 자체는 fold 안에서만 이뤄집니다.
- **enterotype 은 순환입니다.** 이 값 자체가 microbiome 조성으로 정의된 것이라
  microbiome 특징으로 맞추는 것은 부분적으로 자기 자신을 맞추는 일입니다. 표에는
  남겨 두되 그렇게 읽어야 합니다.
- **enterotype 은 세 범주가 크게 불균형합니다.** 3범주 분류로 다루면 다수 범주만
  찍는 모델이 높은 정확도를 받으므로, 여기서는 ET-Firmicutes 대 나머지 이진으로만
  다뤘습니다. 소수 두 범주를 따로 다루려면 표본이 더 필요합니다.
- **KMeans 는 구형 등방 군집을 가정합니다.** 실루엣도 같은 가정을 깔고 있어서,
  층위가 길쭉하거나 밀도가 다르면 k 선택이 틀립니다. 합성 자료에서 3개를 심고 2개를
  고른 것이 그 예입니다.
- **방문(subtype) 정보를 쓰지 않았습니다.** 4회 반복측정을 독립 표본처럼 군집에
  넣었습니다. 궤적 자체를 층위로 삼는 것(예: 혼합효과 군집)이 다음 단계입니다.
