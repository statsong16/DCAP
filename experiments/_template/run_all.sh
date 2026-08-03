#!/usr/bin/env bash
# 합성 데이터로 전체 파이프라인을 점검합니다.
set -e
export PYTHONPATH=src

# python src/make_synthetic.py --out data
# python src/train.py --data data --out runs/main
# python src/evaluate.py --data data --model runs/main
