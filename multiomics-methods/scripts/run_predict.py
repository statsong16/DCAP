"""
run_predict.py — 층위와 임베딩으로 메타데이터를 맞출 수 있는지 잽니다.

분할: 피험자 단위 GroupKFold. fold 마다 전처리·PCA·KMeans 를 train 에서
다시 fit 합니다. 군집 번호는 fold 안에서만 의미가 있으므로 밖으로 나가지
않습니다.

--leaky-split 은 일부러 표본 단위로 나눕니다. 누수의 크기를 주장하지 말고
재보라는 대조군이고, 보고하는 표에는 넣지 마세요.

베이스라인 없이 표를 만들지 않습니다. 항상 함께 도는 것:
  trivial          특징 없이 train 의 평균/사전확률만
  strata_onehot    군집 배정 원-핫 — "층위만으로 얼마나 설명되는가"
  joint_embedding  세 블록을 합친 임베딩
  <block>_pcs      블록 하나만 — 합치는 것이 실제로 이득인지 재는 기준
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mox import qc  # noqa: E402
from mox.evaluate import OutOfFoldCollector, binary_metrics, regression_metrics  # noqa: E402
from mox.integrate import MultiOmicsEmbedder  # noqa: E402
from mox.io import SUBJECT_COL, load_blocks  # noqa: E402
from mox.splits import leaky_folds, subject_folds  # noqa: E402

# (변수, 종류, 양성 정의). 양성이 None 이면 회귀.
TARGETS = [
    ("Gender", "binary", "Female"),
    ("enteroType", "binary", "ET-Firmicutes"),
    ("BMI", "regression", None),
    ("Age", "regression", None),
]

FEATURE_ORDER = ["trivial", "strata_onehot", "joint_embedding",
                 "microbiome_pcs", "metabolome_pcs", "proteome_pcs"]


def models_for(kind: str, seed: int):
    if kind == "binary":
        return {"logreg": make_pipeline(StandardScaler(),
                                        LogisticRegression(max_iter=2000)),
                "gbdt": HistGradientBoostingClassifier(max_iter=200, random_state=seed)}
    return {"ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            "gbdt": HistGradientBoostingRegressor(max_iter=200, random_state=seed)}


def fold_features(blocks, tr, te, n_comp, k, seed):
    """fold 하나의 특징. 학습되는 것은 전부 tr 에서만 fit 합니다."""
    emb = MultiOmicsEmbedder(n_comp, seed).fit(blocks, tr)
    Ztr, Zte = emb.transform(blocks, tr), emb.transform(blocks, te)
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Ztr)
    eye = np.eye(k)
    feats = {"strata_onehot": (eye[km.labels_], eye[km.predict(Zte)]),
             "joint_embedding": (Ztr, Zte)}
    per_tr, per_te = emb.transform_per_block(blocks, tr), emb.transform_per_block(blocks, te)
    for b in per_tr:
        feats[f"{b}_pcs"] = (per_tr[b], per_te[b])
    return feats


def main(data_dir: str, out_dir: str, n_splits: int, n_comp: int, k: int,
         seed: int, leaky: bool, drop_incomplete: float | None):
    os.makedirs(out_dir, exist_ok=True)
    blocks, md = load_blocks(data_dir)

    if drop_incomplete is not None:
        flag = qc.flag_incomplete(blocks, md, drop_incomplete)
        if flag.any():
            print(f"[제외] 블록 결측률 {drop_incomplete:.0%} 초과 표본 {int(flag.sum())}개")
            blocks = {kk: v.loc[~flag].reset_index(drop=True) for kk, v in blocks.items()}
            md = md.loc[~flag].reset_index(drop=True)

    n = len(md)
    if leaky:
        splits = list(leaky_folds(md, n_splits, seed))
        print(f"[분할] 표본 단위 {n_splits}-fold — 누수 대조용, 보고용 아님")
    else:
        splits = list(subject_folds(md, n_splits))
        print(f"[분할] 피험자 단위 GroupKFold {n_splits}-fold "
              f"(피험자 {md[SUBJECT_COL].nunique()}명, 표본 {n}개)")

    oof = OutOfFoldCollector(n)
    for f, (tr, te) in enumerate(splits):
        feats = fold_features(blocks, tr, te, n_comp, k, seed)
        oof.mark(te)
        for tname, kind, pos in TARGETS:
            raw = md[tname].to_numpy()
            y = (raw == pos).astype(int) if kind == "binary" else raw.astype(float)
            for fname, (Xtr, Xte) in feats.items():
                for mname, model in models_for(kind, seed).items():
                    if kind == "binary" and len(np.unique(y[tr])) < 2:
                        continue
                    model.fit(Xtr, y[tr])
                    pred = (model.predict_proba(Xte)[:, 1] if kind == "binary"
                            else model.predict(Xte))
                    oof.add((fname, mname, tname), te, pred)
            # 특징 없는 베이스라인. fold 마다 사전확률이 조금씩 달라서 pooled AUC 가
            # 정확히 0.5 가 아니고 보정기울기도 의미가 없습니다. Brier 만 읽으세요.
            oof.add(("trivial", "-", tname), te, np.full(len(te), y[tr].mean()))
        print(f"[fold {f + 1}/{len(splits)}] train {len(tr)} / test {len(te)}")

    rows = []
    for (fname, mname, tname), p in oof.predictions().items():
        kind, pos = next((kk, pp) for tt, kk, pp in TARGETS if tt == tname)
        raw = md[tname].to_numpy()
        if kind == "binary":
            m = binary_metrics((raw == pos).astype(int), p)
            rows.append({"target": f"{tname}={pos}", "features": fname,
                         "model": mname, **m})
        else:
            rows.append({"target": tname, "features": fname, "model": mname,
                         **regression_metrics(raw.astype(float), p)})

    tab = pd.DataFrame(rows)
    tab["_o"] = tab["features"].map({v: i for i, v in enumerate(FEATURE_ORDER)})
    tab = tab.sort_values(["target", "_o", "model"]).drop(columns="_o")

    for tname, sub in tab.groupby("target", sort=False):
        cols = ["features", "model"] + (["AUC", "Brier", "calib_slope"]
                                        if sub["AUC"].notna().any() else ["R2", "MAE"])
        print(f"\n=== {tname} ===")
        print(sub[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    suffix = "_leaky" if leaky else ""
    tab.to_csv(os.path.join(out_dir, f"results{suffix}.csv"), index=False)
    with open(os.path.join(out_dir, f"config{suffix}.json"), "w") as f:
        json.dump({"seed": seed, "n_splits": n_splits, "k": k,
                   "n_components_per_block": n_comp, "n_samples": int(n),
                   "n_subjects": int(md[SUBJECT_COL].nunique()),
                   "drop_incomplete": drop_incomplete,
                   "split": "sample (leaky)" if leaky else "subject GroupKFold"},
                  f, indent=2)
    print(f"\n[saved] {out_dir}/results{suffix}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/predict")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-components", type=int, default=10)
    ap.add_argument("--k", type=int, default=3, help="run_stratify.py 가 고른 k")
    ap.add_argument("--drop-incomplete", type=float, default=None)
    ap.add_argument("--leaky-split", action="store_true",
                    help="표본 단위 분할. 누수 크기를 보여주는 대조용")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.data, a.out, a.n_splits, a.n_components, a.k, a.seed, a.leaky_split,
         a.drop_incomplete)
