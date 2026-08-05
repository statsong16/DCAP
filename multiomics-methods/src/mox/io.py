"""
io.py — 오믹스 블록을 읽어 표본 순서를 맞춥니다.

저장 규약: CSV 는 **행 = feature, 열 = 표본** 입니다.
오믹스 자료가 원래 그 방향으로 배포되기 때문이고(R 의 assay 행렬, RData 등),
전치는 이 모듈에서 한 번만 합니다. 이후 모든 코드는 samples x features 를 봅니다.

표본 순서를 믿지 않습니다. 블록마다 열 순서가 다를 수 있고, 한 번 어긋나면
그 뒤의 모든 결과가 조용히 틀립니다. 항상 metadata 기준으로 재정렬합니다.
"""

from __future__ import annotations

import os

import pandas as pd

SAMPLE_COL = "sample.ID"
SUBJECT_COL = "metadata.ID"


def load_blocks(data_dir: str, blocks=("microbiome", "metabolome", "proteome"),
                metadata: str = "metadata.csv"):
    """
    CSV 들을 읽어 (blocks, metadata) 로 돌려줍니다.

    blocks[name] 은 samples x features DataFrame 이고, 행 순서는 metadata 와
    정확히 같습니다.
    """
    md = pd.read_csv(os.path.join(data_dir, metadata))
    for col in (SAMPLE_COL, SUBJECT_COL):
        if col not in md.columns:
            raise ValueError(f"metadata 에 '{col}' 컬럼이 필요합니다")
    if md[SAMPLE_COL].duplicated().any():
        raise ValueError(f"{SAMPLE_COL} 에 중복이 있습니다")

    out = {}
    for b in blocks:
        path = os.path.join(data_dir, f"{b}.csv")
        df = pd.read_csv(path, index_col=0).T
        df.index = df.index.astype(str)
        out[b] = _align(df, md[SAMPLE_COL].astype(str), b).astype(float)
    return out, md


def _align(df: pd.DataFrame, sample_ids: pd.Series, name: str) -> pd.DataFrame:
    missing = set(sample_ids) - set(df.index)
    if missing:
        raise ValueError(
            f"{name}: metadata 에 있는 표본 {len(missing)}개가 블록에 없습니다 "
            f"(예: {sorted(missing)[:3]})")
    return df.loc[sample_ids]


def load_rdata(rdata_path: str, object_name: str,
               blocks=("microbiome", "metabolome", "proteome"),
               metadata_key: str = "metadata2"):
    """
    R 의 list 객체 하나를 읽어 load_blocks 와 같은 모양으로 돌려줍니다.

    R 을 설치하지 않아도 되도록 python `rdata` 를 씁니다. dimnames 가 없는
    행렬은 xarray DataArray 로 들어오므로 to_pandas() 로 되돌립니다.
    """
    try:
        import rdata as _rdata
    except ImportError as e:
        raise ImportError("RData 를 읽으려면 `pip install rdata` 가 필요합니다") from e

    obj = _rdata.conversion.convert(
        _rdata.parser.parse_file(rdata_path))[object_name]
    get = {str(k): v for k, v in obj.items()}

    if metadata_key not in get:
        raise ValueError(f"{object_name} 에 '{metadata_key}' 가 없습니다 "
                         f"(있는 것: {sorted(get)})")
    md = pd.DataFrame(get[metadata_key]).reset_index(drop=True)
    sample_ids = md[SAMPLE_COL].astype(str)

    out = {}
    for b in blocks:
        if b not in get:
            raise ValueError(f"{object_name} 에 '{b}' 가 없습니다")
        raw = get[b]
        df = raw.to_pandas() if hasattr(raw, "to_pandas") else pd.DataFrame(raw)
        df = df.astype(float)
        df.columns = df.columns.astype(str)
        # 행=feature, 열=표본 이므로 전치해서 정렬합니다.
        out[b] = _align(df.T, sample_ids, b)
    return out, md


def write_blocks(blocks: dict, md: pd.DataFrame, out_dir: str):
    """load_blocks 가 읽을 수 있는 형식으로 저장합니다 (행=feature, 열=표본)."""
    os.makedirs(out_dir, exist_ok=True)
    for name, df in blocks.items():
        df.T.to_csv(os.path.join(out_dir, f"{name}.csv"))
    md.to_csv(os.path.join(out_dir, "metadata.csv"), index=False)
