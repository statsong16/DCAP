"""
predict.py — 층위/임베딩으로 메타데이터를 맞출 수 있는지 잽니다.

분할 (CLAUDE.md §6):
  피험자(metadata.ID) 단위 GroupKFold 5-fold. 한 사람의 4회 방문은 항상 같은 쪽.
  fold 마다 전처리·PCA·KMeans 를 train 에서 다시 fit 하고 test 에는 transform 만
  적용합니다. 군집 번호는 fold 안에서만 의미가 있으므로 fold 밖으로 나가지 않습니다.
  --leaky-split 을 주면 일부러 표본 단위로 나눕니다. 누수가 얼마나 부풀리는지
  직접 보기 위한 스위치이고, 보고용 수치가 아닙니다.

비교 대상 (베이스라인 없이 표를 만들지 않습니다 — CLAUDE.md §9):
  trivial          train 의 최빈값/평균만. 특징 없음.
  strata_onehot    군집 배정 원-핫. "층위만으로 얼마나 설명되는가"
  joint_embedding  세 블록 PCA 를 이어붙인 30차원
  *_pcs            블록 하나만 쓴 10차원

표적:
  Gender, BMI, Age  — 피험자 상수에 가깝습니다(피험자 내 SD: Age 0.34년, BMI 0.30).
                      그래서 이건 사실상 피험자 69명을 맞추는 문제입니다.
  enteroType        — 주의: 이 값 자체가 microbiome 에서 정의된 것이라
                      microbiome 특징으로 맞추는 것은 부분적으로 순환입니다.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import MultiOmicsEmbedder, load_blocks, subject_folds

# (이름, 종류, 양성 정의) — 양성이 None 이면 회귀
TARGETS = [
    ("Gender", "binary", "Female"),
    ("enteroType", "binary", "ET-Firmicutes"),
    ("BMI", "regression", None),
    ("Age", "regression", None),
]


def calibration_slope(y: np.ndarray, p: np.ndarray) -> float:
    """logit(p) 에 대한 y 의 로지스틱 회귀 기울기. 1이면 잘 보정된 것."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p)).reshape(-1, 1)
    if np.std(x) < 1e-9 or len(np.unique(y)) < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = LogisticRegression(penalty=None, max_iter=1000).fit(x, y)
    return float(lr.coef_[0][0])


def make_models(kind: str, seed: int):
    if kind == "binary":
        return {
            "logreg": make_pipeline(StandardScaler(),
                                    LogisticRegression(max_iter=2000, C=1.0)),
            "gbdt": HistGradientBoostingClassifier(max_iter=200, random_state=seed),
        }
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "gbdt": HistGradientBoostingRegressor(max_iter=200, random_state=seed),
    }


def fold_features(blocks, md, tr, te, n_comp, k, seed):
    """
    fold 하나의 특징을 만듭니다. 학습되는 것은 전부 tr 에서만 fit 합니다.
    반환: {특징이름: (X_train, X_test)}
    """
    emb = MultiOmicsEmbedder(n_comp, seed).fit(blocks, tr)
    Ztr, Zte = emb.transform(blocks, tr), emb.transform(blocks, te)

    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Ztr)
    def onehot(lab):
        return np.eye(k)[lab]
    Str, Ste = onehot(km.labels_), onehot(km.predict(Zte))

    feats = {"strata_onehot": (Str, Ste), "joint_embedding": (Ztr, Zte)}
    per_tr = emb.transform_per_block(blocks, tr)
    per_te = emb.transform_per_block(blocks, te)
    for b in per_tr:
        feats[f"{b}_pcs"] = (per_tr[b], per_te[b])
    return feats


def main(data_dir: str, out_dir: str, n_splits: int, n_comp: int, k: int,
         seed: int, leaky: bool):
    os.makedirs(out_dir, exist_ok=True)
    blocks, md = load_blocks(data_dir)
    n = len(md)

    if leaky:
        # 일부러 잘못 나눕니다. 대조용이며 보고 표에 넣지 않습니다.
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        splits = [(np.setdiff1d(np.arange(n), perm[i::n_splits]), perm[i::n_splits])
                  for i in range(n_splits)]
        print(f"[분할] 표본 단위 {n_splits}-fold — 누수 대조용")
    else:
        splits = list(subject_folds(md, n_splits))
        print(f"[분할] 피험자 단위 GroupKFold {n_splits}-fold "
              f"(피험자 {md['metadata.ID'].nunique()}명, 표본 {n}개)")

    # oof[(feature, model, target)] = 길이 n 의 예측
    oof, filled = {}, np.zeros(n, dtype=bool)
    for f, (tr, te) in enumerate(splits):
        feats = fold_features(blocks, md, tr, te, n_comp, k, seed)
        filled[te] = True
        for tname, kind, pos in TARGETS:
            raw = md[tname].to_numpy()
            y = (raw == pos).astype(int) if kind == "binary" else raw.astype(float)
            for fname, (Xtr, Xte) in feats.items():
                for mname, model in make_models(kind, seed).items():
                    key = (fname, mname, tname)
                    if key not in oof:
                        oof[key] = np.full(n, np.nan)
                    if kind == "binary":
                        if len(np.unique(y[tr])) < 2:
                            continue
                        model.fit(Xtr, y[tr])
                        oof[key][te] = model.predict_proba(Xte)[:, 1]
                    else:
                        model.fit(Xtr, y[tr])
                        oof[key][te] = model.predict(Xte)
            # 특징 없는 베이스라인: train 의 사전확률 / 평균.
            # fold 마다 사전확률이 조금씩 달라서 pooled OOF 의 AUC 가 정확히 0.5 가
            # 아니고 보정기울기도 의미 없는 값이 나옵니다. Brier 만 읽으세요.
            key = ("trivial", "-", tname)
            if key not in oof:
                oof[key] = np.full(n, np.nan)
            oof[key][te] = y[tr].mean()
        print(f"[fold {f + 1}/{len(splits)}] train {len(tr)} / test {len(te)}")
    assert filled.all(), "모든 표본이 정확히 한 번 held-out 이어야 합니다"

    rows = []
    for (fname, mname, tname), p in oof.items():
        kind, pos = next((k_, p_) for t_, k_, p_ in TARGETS if t_ == tname)
        raw = md[tname].to_numpy()
        if kind == "binary":
            y = (raw == pos).astype(int)
            rows.append({"target": f"{tname}={pos}", "features": fname, "model": mname,
                         "AUC": roc_auc_score(y, p),
                         "Brier": brier_score_loss(y, p),
                         "calib_slope": calibration_slope(y, p)})
        else:
            y = raw.astype(float)
            rows.append({"target": tname, "features": fname, "model": mname,
                         "R2": r2_score(y, p), "MAE": mean_absolute_error(y, p)})

    tab = pd.DataFrame(rows)
    order = ["trivial", "strata_onehot", "joint_embedding",
             "microbiome_pcs", "metabolome_pcs", "proteome_pcs"]
    tab["_o"] = tab["features"].map({v: i for i, v in enumerate(order)})
    tab = tab.sort_values(["target", "_o", "model"]).drop(columns="_o")

    for tname, sub in tab.groupby("target", sort=False):
        cols = ["features", "model"] + (
            ["AUC", "Brier", "calib_slope"] if "AUC" in sub and sub["AUC"].notna().any()
            else ["R2", "MAE"])
        print(f"\n=== {tname} ===")
        print(sub[cols].dropna(axis=1, how="all").to_string(
            index=False, float_format=lambda x: f"{x:.3f}"))

    suffix = "_leaky" if leaky else ""
    tab.to_csv(os.path.join(out_dir, f"results{suffix}.csv"), index=False)
    with open(os.path.join(out_dir, f"config{suffix}.json"), "w") as f:
        json.dump({"seed": seed, "n_splits": n_splits, "n_components_per_block": n_comp,
                   "k": k, "split": "sample (leaky)" if leaky else "subject GroupKFold",
                   "n_samples": int(n),
                   "n_subjects": int(md["metadata.ID"].nunique())}, f, indent=2)
    print(f"\n[saved] {out_dir}/results{suffix}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/predict")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-components", type=int, default=10)
    ap.add_argument("--k", type=int, default=3, help="stratify.py 가 고른 k")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--leaky-split", action="store_true",
                    help="표본 단위 분할. 누수 효과를 보여주는 대조용")
    a = ap.parse_args()
    main(a.data, a.out, a.n_splits, a.n_components, a.k, a.seed, a.leaky_split)
