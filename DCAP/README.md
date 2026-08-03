# Dementia Cohort Analysis

[![CI](https://github.com/STATSONG/dementia-cohort-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/STATSONG/dementia-cohort-analysis/actions/workflows/ci.yml)

치매 코호트 자료에 대한 **통계 방법론 및 표현학습 실험** 저장소입니다.
모든 파이프라인은 합성 데이터만으로 끝까지 재현되며, 참여자 자료는 포함하지 않습니다.

**Statistical methodology and representation-learning experiments for dementia
cohort data. Every pipeline reproduces end-to-end on synthetic data — no
participant data is included, and no data access agreement is needed to run it.**

---

## 구성

```
experiments/          실험 하나 = 디렉터리 하나. 각자 README·실행 명령·결과 표를 가집니다.
  01-microbiome-tokenization/   조성 자료를 트랜스포머 토큰으로 바꾸는 전략 비교
  _template/                    새 실험을 시작하는 틀
notes/                읽은 논문과 설계 판단 기록 — "왜 그렇게 했는가"
CLAUDE.md             이 저장소의 작업 규칙
```

공용 유틸리티 디렉터리는 두지 않았습니다. 실험이 하나뿐인 지금 만들면
단일 사용처를 위한 추상화가 되기 때문입니다(`CLAUDE.md` §2). 두 번째 실험에서
같은 코드가 반복되면 그때 `common/` 으로 올립니다.

---

## 실행

```bash
git clone https://github.com/STATSONG/dementia-cohort-analysis
cd dementia-cohort-analysis
make setup          # pip install -r requirements.txt
make demo           # 합성 데이터로 01번 실험 전체 파이프라인
```

실제 데이터로 돌리려면 각 실험 README의 "데이터 접근" 절을 참고하세요.
`data/` 는 git-ignore 되어 있습니다.

---

## 실험 목록

| # | 주제 | 상태 | 요약 |
|---|---|---|---|
| 01 | [마이크로바이옴 토큰화](experiments/01-microbiome-tokenization/) | 합성 데이터 검증 완료 | 마스킹 복원 사전학습 · taxa↔pathway 대조정렬 · 계통수 어텐션 마스크 |
| 02 | 영상 지표 토큰화 (예정) | — | OASIS-3 FreeSurfer/PUP 파생 지표 기반 |

---

## 이 저장소가 지키는 것

`CLAUDE.md` 에 전문이 있습니다. 핵심 다섯 가지만 적으면:

1. **계산하지 않은 숫자는 쓰지 않습니다.** README의 모든 수치는 이 저장소의 코드를 실제로 실행해 나온 값입니다.
2. **분할은 study 단위입니다.** 여러 연구를 모은 자료에서 표본 단위 분할은 배치효과를 학습해 성능을 부풀립니다.
3. **참여자 자료는 커밋하지 않습니다.** 합성 데이터로 전체가 도는 것이 저장소의 조건입니다.
4. **시드와 버전을 기록합니다.** 명령 하나로 보고된 표가 재생성됩니다.
5. **베이스라인에 진 결과도 남깁니다.** 이긴 것만 싣는 표는 결과가 아닙니다.

---

## 라이선스

코드는 MIT. 데이터는 각 제공기관의 이용 조건을 따릅니다.
