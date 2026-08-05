"""
run_missing_data.py — 결측 대치 방식이 결과를 어떻게 바꾸는지 재 봅니다.

주장하지 않고 측정합니다. 세 가지를 같은 자료에 걸고 두 가지를 잽니다.

  방식
    median_all      모든 블록을 train 중앙값으로 대치. 흔한 기본값이고,
                    결측 기전을 구분하지 않습니다.
    by_mechanism    mox 기본값. 블록의 결측 기전에 맞춰 고릅니다
                    (log_abundance -> 좌측중도절단, intensity -> 중앙값).
    by_mechanism
      +drop         위에 더해, 블록 결측률이 큰 표본을 아예 뺍니다.

  측정 1 — 검출한계 대치의 편향 (proteome)
    합성 자료에서는 결측 자리의 참값이 LOD 아래였음을 압니다. 대치값이 LOD
    위에 있으면 그만큼 위로 끌어올린 것이고, 그 차이가 곧 편향입니다.

  측정 2 — 블록 결측 표본의 위치 왜곡 (metabolome)
    대치는 결측이 많은 표본을 중심으로 끌어당깁니다. 결측 피험자가 대사체
    PCA 공간에서 중심까지 거리 기준 몇 퍼센타일에 있는지 봅니다. 낮을수록
    "인위적으로 평균적인 표본"이 되었다는 뜻입니다.

측정 2 가 방식만으로는 고쳐지지 않는다는 점이 핵심입니다. 대사체의 결측은
검출한계가 아니라 검체/배치 실패라서, 대치를 바꾸는 것으로는 해결되지
않고 표본을 빼야 합니다. 기전을 먼저 보고 처리를 고르라는 이유입니다.

정답(LOD, 결측 피험자)을 알아야 채점할 수 있으므로 합성 자료 전용입니다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mox import qc  # noqa: E402
from mox.integrate import MultiOmicsEmbedder  # noqa: E402
from mox.io import SUBJECT_COL, load_blocks  # noqa: E402
from mox.preprocess import BlockPrep  # noqa: E402

# 이름 -> (BlockPrep 에 강제할 kwargs, 불완전 표본을 뺄지, 참 LOD 를 넘길지)
# 마지막 방식은 실제 LOD 값을 아는 경우입니다. Olink 처럼 LOD 가 함께 배포되는
# 자료라면 분위수로 근사하지 말고 그 값을 그대로 넘기는 것이 맞습니다.
STRATEGIES = {
    "median_all":        ({"na_strategy": "median"}, False, False),
    "by_mechanism":      ({}, False, False),
    "by_mechanism+drop": ({}, True, False),
    "true_lod":          ({}, False, True),
}


def imputation_bias(proteome: pd.DataFrame, lod: float, prep_kwargs: dict) -> dict:
    """proteome 결측 자리에 채워 넣는 값이 참 LOD 대비 어디에 있는지."""
    X = proteome.to_numpy(dtype=float)
    prep = BlockPrep("log_abundance", **prep_kwargs).fit(X)
    kept_na = np.isnan(X)[:, prep.keep_]
    fill = np.broadcast_to(prep.fill_, kept_na.shape)[kept_na]
    if fill.size == 0:
        return {"n_imputed": 0, "mean_fill": float("nan"),
                "bias_above_lod": float("nan"), "frac_above_lod": float("nan")}
    return {"n_imputed": int(fill.size), "mean_fill": float(fill.mean()),
            "bias_above_lod": float(fill.mean() - lod),
            "frac_above_lod": float((fill > lod).mean())}


def centroid_percentile(blocks: dict, affected: np.ndarray, prep_kwargs: dict,
                        seed: int) -> dict:
    """대사체 PCA 공간에서 중심까지 거리의 백분위. 낮을수록 중심에 끌려간 것."""
    if not affected.any():
        return {"n_affected": 0, "affected_pct": float("nan"),
                "others_pct": float("nan")}
    emb = MultiOmicsEmbedder(10, seed, prep_kwargs=prep_kwargs).fit(blocks)
    Z = emb.transform_per_block(blocks)["metabolome"]
    d = np.linalg.norm(Z - Z.mean(axis=0), axis=1)
    pct = np.array([(d < x).mean() * 100 for x in d])
    return {"n_affected": int(affected.sum()),
            "affected_pct": float(np.median(pct[affected])),
            "others_pct": float(np.median(pct[~affected]))}


def main(data_dir: str, out_dir: str, seed: int, max_missing: float):
    os.makedirs(out_dir, exist_ok=True)
    blocks, md = load_blocks(data_dir)

    lod_path = os.path.join(data_dir, "proteome_lod.csv")
    if not os.path.exists(lod_path):
        raise SystemExit("proteome_lod.csv 가 없습니다. 정답을 아는 합성 자료 전용입니다 "
                         "(scripts/make_synthetic.py 로 생성)")
    lod = float(pd.read_csv(lod_path)["lod"].iloc[0])

    flag = qc.flag_incomplete(blocks, md, max_missing)
    affected_subjects = sorted(map(str, set(md.loc[flag, SUBJECT_COL])))
    print(f"결측률 {max_missing:.0%} 초과 표본 {int(flag.sum())}개 · "
          f"피험자 {len(affected_subjects)}명 ({', '.join(affected_subjects)})\n")

    lod_vec = pd.read_csv(lod_path)["lod"].to_numpy()

    rows = []
    for name, (prep_kwargs, drop, use_true_lod) in STRATEGIES.items():
        b, m = blocks, md
        if drop:
            keep = ~flag
            b = {k: v.loc[keep].reset_index(drop=True) for k, v in blocks.items()}
            m = md.loc[keep].reset_index(drop=True)
        affected = m[SUBJECT_COL].astype(str).isin(affected_subjects).to_numpy()

        prot_kwargs = {**prep_kwargs, "lod": lod_vec} if use_true_lod else prep_kwargs
        # 참 LOD 는 proteome 전용 벡터라 다른 블록에 그대로 넘길 수 없습니다.
        # 그래서 이 방식은 측정 1 만 냅니다.
        loc = ({"n_affected": int(affected.sum()), "affected_pct": float("nan"),
                "others_pct": float("nan")} if use_true_lod
               else centroid_percentile(b, affected, prep_kwargs, seed))
        rows.append({"strategy": name, "n_samples": len(m),
                     **imputation_bias(b["proteome"], lod, prot_kwargs), **loc})

    tab = pd.DataFrame(rows)
    print(f"측정 1 — proteome 검출한계 대치의 편향 (참 LOD = {lod:.2f})")
    print(tab[["strategy", "n_imputed", "mean_fill", "bias_above_lod",
               "frac_above_lod"]].to_string(index=False,
                                            float_format=lambda x: f"{x:.3f}"))
    print("\n측정 2 — 대사체 결측 피험자의 위치 (중심까지 거리 백분위)")
    print(tab[["strategy", "n_samples", "n_affected", "affected_pct",
               "others_pct"]].to_string(index=False,
                                        float_format=lambda x: f"{x:.1f}"))

    tab.to_csv(os.path.join(out_dir, "missing_data.csv"), index=False)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"seed": seed, "max_missing": max_missing, "lod": lod,
                   "affected_subjects": affected_subjects}, f, indent=2)
    print(f"\n[saved] {out_dir}/missing_data.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/missing")
    ap.add_argument("--max-missing", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.data, a.out, a.seed, a.max_missing)
