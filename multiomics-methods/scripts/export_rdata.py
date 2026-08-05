"""
export_rdata.py — R 의 list 객체 하나를 파이프라인이 읽는 CSV 로 내보냅니다.

기대하는 구조 (이름은 인자로 바꿀 수 있습니다):

  <object>$microbiome  features x samples
  <object>$metabolome  features x samples
  <object>$proteome    features x samples
  <object>$metadata2   samples x covariates   (sample.ID, metadata.ID 필수)

출력은 git-ignore 되는 곳에만 쓰세요. 기본 경로 data/ 가 그렇게 되어 있습니다.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mox.io import load_rdata, write_blocks  # noqa: E402


def main(rdata: str, out_dir: str, object_name: str, blocks, metadata_key: str):
    b, md = load_rdata(rdata, object_name, tuple(blocks), metadata_key)
    write_blocks(b, md, out_dir)
    for name, df in b.items():
        X = df.to_numpy(dtype=float)
        print(f"[export] {name:11s} {X.shape[1]:5d} feature x {X.shape[0]:4d} 표본  "
              f"결측 {np.isnan(X).mean():.4f}")
    print(f"[export] metadata    {md.shape[0]:5d} 표본 x {md.shape[1]:2d} 변수  "
          f"피험자 {md['metadata.ID'].nunique()}명")
    print(f"[export] -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdata", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--object", default="multi_omics_set")
    ap.add_argument("--blocks", nargs="+",
                    default=["microbiome", "metabolome", "proteome"])
    ap.add_argument("--metadata-key", default="metadata2")
    a = ap.parse_args()
    main(a.rdata, a.out, a.object, a.blocks, a.metadata_key)
