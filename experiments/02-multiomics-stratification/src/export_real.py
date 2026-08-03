"""
export_real.py — multi_omics_set.RData 를 파이프라인이 읽는 CSV 4개로 내보냅니다.

  multi_omics_set$microbiome  (1350 x 276)  행=MSP,        열=표본
  multi_omics_set$metabolome  ( 413 x 276)  행=대사체,     열=표본
  multi_omics_set$proteome    ( 770 x 276)  행=단백질,     열=표본
  multi_omics_set$metadata2   ( 276 x  10)

방향(행=feature, 열=표본)을 그대로 유지합니다. 전치는 data.py 가 합니다.

출력은 참여자 자료에서 파생된 것이므로 git-ignore 되는 곳에만 쓰세요.
기본 출력 경로 data/ 는 저장소 .gitignore 에 들어 있습니다 (CLAUDE.md §7).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

EXPECTED = ("microbiome", "metabolome", "proteome", "metadata2")


def main(rdata_path: str, out_dir: str):
    try:
        import rdata
    except ImportError:
        raise SystemExit("rdata 가 필요합니다: pip install -r requirements.txt")

    os.makedirs(out_dir, exist_ok=True)
    parsed = rdata.parser.parse_file(rdata_path)
    obj = rdata.conversion.convert(parsed)["multi_omics_set"]

    missing = [k for k in EXPECTED if k not in {str(x) for x in obj.keys()}]
    if missing:
        raise SystemExit(f"multi_omics_set 에 없는 항목: {missing}")
    get = {str(k): v for k, v in obj.items()}.get

    md = pd.DataFrame(get("metadata2")).reset_index(drop=True)
    sample_ids = md["sample.ID"].astype(str).tolist()

    for name in ("microbiome", "metabolome", "proteome"):
        blk = get(name)
        # microbiome 은 dimnames 가 없어 xarray DataArray 로 들어옵니다.
        df = blk.to_pandas() if hasattr(blk, "to_pandas") else pd.DataFrame(blk)
        df = df.astype(float)
        if list(df.columns.astype(str)) != sample_ids:
            if set(df.columns.astype(str)) == set(sample_ids):
                df = df.loc[:, sample_ids]
            else:
                raise SystemExit(f"{name}: 열 이름이 metadata2 의 sample.ID 와 다릅니다")
        df.columns = sample_ids
        df.index = df.index.astype(str)
        df.to_csv(os.path.join(out_dir, f"{name}.csv"))
        print(f"[export] {name:11s} {df.shape[0]:5d} x {df.shape[1]:4d}  "
              f"NA={np.isnan(df.to_numpy()).mean():.4f}")

    md.to_csv(os.path.join(out_dir, "metadata.csv"), index=False)
    print(f"[export] metadata    {md.shape[0]:5d} x {md.shape[1]:4d}  "
          f"subjects={md['metadata.ID'].nunique()}")
    print(f"[export] -> {out_dir}  (참여자 파생 자료. 커밋하지 마세요)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdata", default="../../multi_omics_set.RData")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    main(a.rdata, a.out)
