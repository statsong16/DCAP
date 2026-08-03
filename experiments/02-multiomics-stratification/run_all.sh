#!/usr/bin/env bash
# 합성 데이터로 전체 파이프라인을 한 번에 점검합니다.
# 실제 자료로 돌리려면 아래 DATA 를 real 로 바꾸고 export_real.py 를 먼저 실행하세요.
set -e
export PYTHONPATH=src

SEED=${SEED:-0}

python src/make_synthetic.py --out data --seed "$SEED"
python src/stratify.py  --data data --out runs/stratify --seed "$SEED"

# 군집 수는 stratify 가 실루엣으로 고른 값을 그대로 씁니다.
K=$(python -c "import json;print(json.load(open('runs/stratify/summary.json'))['k'])")
echo "[run_all] stratify 가 고른 k = $K"

python src/predict.py   --data data --out runs/predict  --seed "$SEED" --k "$K"

# 누수 대조: 같은 파이프라인을 표본 단위로 나눠 돌립니다. 보고용 수치가 아닙니다.
python src/predict.py   --data data --out runs/predict  --seed "$SEED" --k "$K" --leaky-split
