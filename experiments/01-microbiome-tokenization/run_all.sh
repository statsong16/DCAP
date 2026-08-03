#!/usr/bin/env bash
# 합성 데이터로 전체 파이프라인을 한 번에 점검합니다.
set -e
export PYTHONPATH=src

python src/make_synthetic.py --out data
python src/pretrain_masked.py --data data --out runs/masked        --epochs 15
python src/pretrain_masked.py --data data --out runs/masked_phylo  --epochs 15 --use-phylo
python src/crossmodal.py      --data data --out runs/crossmodal    --epochs 8
python src/downstream.py      --data data \
    --encoder runs/masked/encoder.pt \
    --embeddings runs/crossmodal/embeddings.npz \
    --label study_condition --positive case
