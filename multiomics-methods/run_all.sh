#!/usr/bin/env bash
# 합성 자료로 전체 파이프라인을 한 번에 돌립니다.
# 실제 자료로 돌리려면 먼저:
#   python scripts/export_rdata.py --rdata /경로/multi_omics_set.RData --out data
# 를 실행하고 아래 make_synthetic 줄만 지우세요.
set -e
export PYTHONPATH=src

SEED=${SEED:-0}
DROP=${DROP:-0.2}          # 블록 결측률이 이 값을 넘는 표본은 제외

python -m pytest -q

# 1. 합성 자료 (실제 자료의 구조와 결측 병리를 재현)
python scripts/make_synthetic.py --out data --seed "$SEED"

# 2. 자료를 먼저 본다 — 상수 변수, 반복측정 구조, 결측 기전
python scripts/run_describe.py --data data --out runs/describe --seed "$SEED"

# 3. 결측 처리 방식이 결과를 어떻게 바꾸는지 (합성 자료 전용: 정답을 알아야 채점됨)
python scripts/run_missing_data.py --data data --out runs/missing --seed "$SEED"

# 4. 층위화
python scripts/run_stratify.py --data data --out runs/stratify \
    --seed "$SEED" --drop-incomplete "$DROP"

# 5. 층위/임베딩으로 메타데이터 예측. k 는 4번이 고른 값을 그대로 씁니다.
K=$(python -c "import json;print(json.load(open('runs/stratify/summary.json'))['k'])")
echo "[run_all] run_stratify 가 고른 k = $K"
python scripts/run_predict.py --data data --out runs/predict \
    --seed "$SEED" --k "$K" --drop-incomplete "$DROP"

# 6. 누수 대조 — 분할만 표본 단위로 바꿉니다. 보고용 아님.
python scripts/run_predict.py --data data --out runs/predict \
    --seed "$SEED" --k "$K" --drop-incomplete "$DROP" --leaky-split

# 7. 연관 검정이 시드 하나의 우연인지 확인 (시드 5개)
python scripts/run_seed_check.py --seeds 0 1 2 3 4 --out runs/seed_check
