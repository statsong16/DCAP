# multiomics-methods

다중오믹스 코호트 자료를 **누수 없이** 분석하기 위한 방법론 패키지.

전처리 · 결합 · 층위화 · 평가를 import 해서 쓰는 모듈로 만들고, 각 설계 선택의
근거를 `docs/` 에 남겼습니다. 자료 접근 없이 클론 하나로 전부 재현됩니다.

```bash
pip install -r requirements.txt
make test        # 단위 테스트 57개
make demo        # 합성 자료로 전체 파이프라인 (약 2분)
```

---

## 이 패키지가 담고 있는 네 가지 판단

| | 판단 | 모듈 | 문서 |
|---|---|---|---|
| 1 | 분할은 **피험자(그룹) 단위**로. 반복측정 자료에서 표본 단위 분할은 표현형이 아니라 사람을 외우게 만든다 | `mox.splits` | [leakage.md](docs/leakage.md) |
| 2 | 블록마다 값의 성격이 다르므로 변환도 다르다. 파라미터를 학습하는 전처리는 **train fold 에서만** fit 한다 | `mox.preprocess` | [methodology.md](docs/methodology.md) |
| 3 | 무작위 결측 · 검출한계 결측 · 배치 실패는 **다른 문제**다. 중앙값으로 다 메우면 두 번째를 위로 끌어올리고 세 번째를 감춘다 | `mox.qc` | [missing-data.md](docs/missing-data.md) |
| 4 | 판별력만 보지 않는다. **보정**을 함께 본다 | `mox.evaluate` | [methodology.md](docs/methodology.md) |

---

## 구성

```
src/mox/            방법론 모듈 (import 해서 쓰는 층)
  io.py             블록 로딩 · 표본 순서 정렬 · RData 읽기
  qc.py             결측 기전 진단 — 무작위 / 검출한계 / 배치 실패
  preprocess.py     블록별 값 변환 + 결측 대치 (fit/transform)
  splits.py         피험자 단위 GroupKFold + 누수 검사
  integrate.py      블록별 PCA → 결합 임베딩
  cluster.py        층위화 + 안정성 + 그룹 단위 연관 검정
  evaluate.py       AUC · Brier · 보정기울기 · OOF 수집기
  tokenize.py       조성 자료 → 트랜스포머 토큰 (CLR · 순위/구간/연속 · 계통 마스크)

scripts/            실행 진입점
  make_synthetic.py     실제 자료의 구조와 결측 병리를 재현한 합성 자료
  export_rdata.py       RData → CSV
  run_describe.py       기술 통계 · 그림 · 결측 진단
  run_missing_data.py   결측 처리 방식 비교 (합성 자료 전용)
  run_stratify.py       층위화
  run_predict.py        메타데이터 예측 (+ --leaky-split 누수 대조)
  run_seed_check.py     연관 검정이 시드 하나의 우연인지 확인

R/descriptive.R     기술 통계의 R 판
docs/               왜 그렇게 했는가
tests/              57개. 누수 방지 테스트가 핵심
```

---

## 쓰는 법

```python
from mox.io import load_blocks
from mox import qc
from mox.integrate import MultiOmicsEmbedder
from mox.splits import subject_folds

blocks, md = load_blocks("data")

print(qc.report(blocks, md))                    # 1. 결측이 왜 생겼는지 먼저 본다
keep = ~qc.flag_incomplete(blocks, md, 0.20)    # 2. 대치할 수 없는 표본을 뺀다

for train_idx, test_idx in subject_folds(md, 5):        # 3. 피험자 단위 분할
    emb = MultiOmicsEmbedder(10).fit(blocks, train_idx) # 4. train 에서만 fit
    Z_train = emb.transform(blocks, train_idx)
    Z_test = emb.transform(blocks, test_idx)            # 5. test 는 transform 만
```

블록 이름이 `microbiome`/`metabolome`/`proteome` 이 아니면 `kind_map` 으로
값의 성격을 알려 주세요 (`compositional` · `intensity` · `log_abundance`).

---

## 결과

아래는 전부 **합성 자료**(시드 0)에서 나온 값입니다. 실제 코호트 자료로 돌린
수치는 이 저장소에 없습니다.

| 항목 | 값 |
|---|---|
| 자료 | 피험자 69 × 방문 4 = 표본 276, 잠재 층위 3개 |
| 결측 처리 후 | 표본 268 · 피험자 67 (블록 결측 피험자 2명 제외) |
| 시드 | 0 (모든 스크립트 `--seed 0`) |
| 실행 시간 | 105–115초 (CPU, 두 번 측정) |
| 환경 | Python 3.11.15, numpy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0, scipy 1.17.1, matplotlib 3.11.1 |

### 누수 대조 — 이 저장소에서 제일 확실한 결과

같은 자료, 같은 전처리, 같은 모델. **분할만** 바꿨습니다.
합성 자료에는 성별 신호도 연령 신호도 심지 않았습니다.

| 표적 | 특징 | 모델 | 피험자 단위 | 표본 단위 |
|---|---|---|---|---|
| Gender (AUC) | proteome_pcs | gbdt | 0.531 | **0.957** |
| Gender (AUC) | joint_embedding | gbdt | 0.508 | **0.933** |
| Age (R²) | proteome_pcs | gbdt | −0.185 | **0.682** |
| BMI (R²) | proteome_pcs | gbdt | −0.205 | **0.726** |

신호가 없는 표적에서 AUC 0.96 이 나옵니다. 전부 같은 사람의 다른 방문을
맞춘 것입니다.

### 결측 처리 (참 LOD = −1.50)

| 방식 | 평균 대치값 | LOD 대비 편향 | 결측 피험자 위치(백분위) |
|---|---|---|---|
| 모든 블록 중앙값 | −0.428 | **+1.072** | 1.3 |
| 기전별 (5% 분위수) | −1.266 | +0.234 | 1.3 |
| 기전별 + 표본 제외 | −1.267 | +0.233 | (제외) |
| 실제 LOD 사용 | −1.500 | **0.000** | — |

두 가지가 보입니다. **검출한계 편향은 대치 방식으로 줄어듭니다**(1.07 → 0.23,
실제 LOD 를 쓰면 0). 반면 **배치 실패로 생긴 결측은 대치를 바꿔도 그대로**
입니다(1.3 퍼센타일 → 1.3). 그건 표본을 빼야 고쳐집니다.

### 층위화

| 항목 | 값 |
|---|---|
| 고른 k (실루엣) | 3 (정답도 3) |
| 군집 크기 | 55 / 82 / 131 |
| 실루엣 | 0.082 |
| 부트스트랩 ARI | 0.710 ± 0.190 (피험자 리샘플 20회) |
| 피험자 일관성 | 0.970 |
| 정답 층위와의 ARI | 0.979 |
| BMI R² 상한 | 0.205 |

신호를 20% 줄이면(`--effect-scale 1.0`) 실루엣이 k=6 까지 단조 증가하고
경고가 뜨며 부트스트랩 ARI 가 0.24, 정답과의 ARI 가 0.393 으로 떨어집니다.
진단이 "구조 있음/없음"을 실제로 구분한다는 뜻입니다.

### 메타데이터 예측 (피험자 단위 GroupKFold 5-fold, pooled OOF)

| 표적 | 특징 | 모델 | R² | MAE |
|---|---|---|---|---|
| BMI | trivial | – | −0.065 | 2.582 |
| BMI | strata_onehot | ridge | −0.009 | 2.442 |
| BMI | joint_embedding | ridge | 0.049 | 2.403 |
| **BMI** | **proteome_pcs** | **ridge** | **0.088** | **2.372** |

전체 표(6 특징 × 2 모델 × 4 표적, 누수 대조 포함)는
`runs/predict/results.csv` 에 있습니다.

**해석.** BMI 최고가 0.088 인데 상한이 0.205 이므로 있는 신호의 43% 를
건졌습니다. Gender·Age·enterotype 은 전부 우연 수준이고, 그 신호를 심지
않았으므로 **정확히 그래야 하는 결과**입니다.

**회귀 10칸 전부 ridge 가 gbdt 를 이겼습니다.** 표본 268개(실질 피험자 67명)에
부스팅을 쓸 근거가 없다는 뜻입니다.

**보정기울기는 전부 1보다 한참 작습니다.** 가장 큰 값이 0.398 입니다.
AUC 만 보고 넘어가면 안 되는 이유입니다.

### 연관 검정은 시드 하나로 결론내지 않는다

층위와 실제로 연결한 변수는 **BMI 하나뿐**입니다. 시드 5개 결과:

| 변수 | q<0.05 인 시드 | |
|---|---|---|
| BMI | **5/5** | 심어 둔 신호 |
| Age | 1/5 | 거짓양성 |
| Gender | 1/5 | 거짓양성 |
| enteroType | 0/5 | |

BH 보정을 했는데도 심지 않은 변수에서 20칸 중 2칸(10%)이 유의하게 나왔습니다.
하필 시드 0 이 그런 판이라 Age(q=0.041)와 Gender(q=0.029)가 걸립니다.
**시드 0 의 표만 봤다면 틀린 문장을 두 개 쓰게 됩니다.**

---

## 실제 자료로 돌리기

```bash
python scripts/export_rdata.py --rdata /경로/multi_omics_set.RData --out data
python scripts/run_describe.py  --data data --out runs/describe
python scripts/run_stratify.py  --data data --out runs/stratify --drop-incomplete 0.2
python scripts/run_predict.py   --data data --out runs/predict  --k 3 --drop-incomplete 0.2
```

`run_missing_data.py` 와 `run_seed_check.py` 는 정답을 알아야 채점되므로 합성
자료 전용입니다.

`data/` 와 `runs/` 는 git-ignore 되어 있습니다. 원자료도 그 파생물도 커밋하지
않습니다.

---

## 한계

- **위 수치는 전부 합성 자료입니다.** 실제 코호트 자료로 돌린 결과는 여기 없습니다.
- **k 를 fold 밖에서 골랐습니다.** `run_predict.py` 는 `run_stratify.py` 가 전체
  자료 실루엣으로 고른 k 를 받아 쓰므로, 이 하이퍼파라미터 하나에 약한 선택
  누수가 있습니다. 전처리·PCA·KMeans 적합 자체는 fold 안에서만 이뤄집니다.
- **KMeans 는 구형 등방 군집을 가정합니다.** 실루엣도 같은 가정을 깔고 있어서
  층위가 길쭉하거나 밀도가 다르면 k 선택이 틀립니다.
- **궤적을 쓰지 않았습니다.** 반복측정을 독립 표본처럼 넣었습니다. 변화 자체를
  층위로 삼는 것(혼합효과 군집 등)은 다음 단계입니다.
- **배치 보정이 없습니다.** 플레이트·런 정보가 있는 자료라면 ComBat 계열이
  필요할 수 있습니다.
- **트랜스포머 학습 코드는 없습니다.** `mox.tokenize` 는 토큰화까지만 합니다.
  학습 루프는 방법론이 아니라 실험이라, 검증하지 못한 코드를 패키지에 넣지
  않으려고 뺐습니다.
- **분류에서 소수 범주를 따로 다루지 않았습니다.** enterotype 은 세 범주가 크게
  불균형해서 ET-Firmicutes 대 나머지 이진으로만 다뤘습니다.

## 라이선스

MIT.
