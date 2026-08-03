"""
downstream.py — 학습된 표현이 실제로 쓸모가 있는지 검증하는 층.

여기가 통계 담당자의 자리입니다. 반드시 지키는 세 가지:

 1) study 단위 그룹 교차검증
    cMD 는 여러 연구를 모아 놓은 자료입니다. 같은 연구가 학습·검증에
    걸치면 배치효과를 학습해서 성능이 부풀려집니다. GroupKFold 를 씁니다.

 2) 베이스라인과 나란히 비교
    "특화 파운데이션 모델이 잘 만든 지도학습 베이스라인을 못 이긴다"는
    지적이 반복해서 나옵니다. CLR + 로지스틱, CLR + 부스팅을 항상 같이 돌립니다.

 3) AUC 만 보지 않기
    예방 서비스에서는 예측확률이 실제 발생률과 맞는지(보정)가 더 중요합니다.
    Brier score 와 보정 기울기를 함께 보고합니다.

실행 예:
  python src/downstream.py --data data --encoder runs/masked/encoder.pt \
      --embeddings runs/crossmodal/embeddings.npz --label study_condition --positive case
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from data import MicrobiomeTokenDataset, load_tables
from models import TaxaTransformer


# ---------------------------------------------------------------------
# 임베딩 추출
# ---------------------------------------------------------------------
@torch.no_grad()
def encoder_embeddings(ckpt_path: str, ds: MicrobiomeTokenDataset,
                       device: str = "cpu", batch_size: int = 64) -> np.ndarray:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck["args"]
    enc = TaxaTransformer(ck["vocab_size"], d_model=a["d_model"], n_heads=a["n_heads"],
                          n_layers=a["n_layers"], max_len=a["max_len"],
                          value_mode=a["value_mode"], n_bins=a["n_bins"]).to(device)
    enc.load_state_dict(ck["encoder"])
    enc.eval()

    out = []
    for batch in DataLoader(ds, batch_size=batch_size, shuffle=False):
        h = enc(batch["ids"].to(device), batch["ranks"].to(device),
                batch["vals"].to(device), batch["pad_mask"].to(device))
        out.append(enc.pool(h, batch["pad_mask"].to(device)).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------
def calibration_slope(y: np.ndarray, p: np.ndarray) -> float:
    """
    보정 기울기. logit(p) 를 설명변수로 y 를 회귀했을 때의 기울기.
      1에 가까우면 잘 보정됨 / <1 이면 과신 / >1 이면 과소평가
    """
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return float("nan")
    lr = LogisticRegression(penalty=None, max_iter=1000).fit(z, y)
    return float(lr.coef_[0][0])


def grouped_eval(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                 model_fn, n_splits: int | None = None) -> dict:
    """study 단위 GroupKFold 로 out-of-fold 예측을 모아 평가합니다."""
    n_groups = len(np.unique(groups))
    k = n_splits or min(5, n_groups)
    if k < 2:
        raise ValueError("연구가 하나뿐이라 그룹 교차검증이 불가능합니다.")
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = model_fn().fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(oof)
    return {
        "n_folds": k,
        "auc": float(roc_auc_score(y[ok], oof[ok])),
        "brier": float(brier_score_loss(y[ok], oof[ok])),
        "calibration_slope": calibration_slope(y[ok], oof[ok]),
    }


def logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


def gbdt():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--encoder", default=None, help="runs/masked/encoder.pt")
    ap.add_argument("--embeddings", default=None, help="runs/crossmodal/embeddings.npz")
    ap.add_argument("--label", default="study_condition")
    ap.add_argument("--positive", default="case")
    ap.add_argument("--group", default="study_name")
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--value-mode", default="bin")
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--out", default="runs/downstream_report.json")
    args = ap.parse_args()

    taxa, pathway, meta, _ = load_tables(args.data)
    y = (meta[args.label].astype(str) == args.positive).to_numpy().astype(int)
    groups = meta[args.group].astype(str).to_numpy()
    print(f"표본 {len(y)}개 | 양성 {y.sum()}개 ({y.mean():.1%}) | 연구 {len(np.unique(groups))}개")
    if y.sum() < 10 or (len(y) - y.sum()) < 10:
        print("경고: 사건 수가 너무 적습니다. AUC 해석에 주의하세요.")

    ds = MicrobiomeTokenDataset(taxa, args.max_len, args.value_mode, args.n_bins)

    reps: dict[str, np.ndarray] = {
        "baseline_CLR": ds.clr,
    }
    if pathway is not None:
        ds_p = MicrobiomeTokenDataset(pathway, args.max_len, args.value_mode, args.n_bins)
        reps["baseline_CLR_taxa+pathway"] = np.hstack([ds.clr, ds_p.clr])

    if args.encoder and os.path.exists(args.encoder):
        reps["masked_pretrain_embedding"] = encoder_embeddings(args.encoder, ds)
    if args.embeddings and os.path.exists(args.embeddings):
        z = np.load(args.embeddings, allow_pickle=True)
        reps["crossmodal_embedding"] = np.hstack([z["taxa"], z["pathway"]])

    rows = []
    for name, X in reps.items():
        for clf_name, fn in [("logreg", logreg), ("gbdt", gbdt)]:
            r = grouped_eval(X, y, groups, fn)
            r.update({"representation": name, "classifier": clf_name, "n_features": X.shape[1]})
            rows.append(r)
            print(f"{name:32s} {clf_name:7s} "
                  f"AUC {r['auc']:.3f} | Brier {r['brier']:.3f} | 보정기울기 {r['calibration_slope']:.2f}")

    df = pd.DataFrame(rows)[["representation", "classifier", "n_features",
                             "n_folds", "auc", "brier", "calibration_slope"]]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_json(args.out, orient="records", indent=2)
    print(f"\n표 저장 -> {args.out}")
    print("\n해석 요령: 임베딩이 baseline_CLR 을 이기지 못하면 그것도 결과입니다.")
    print("           제안서에는 이긴 경우만이 아니라 진 경우도 적어야 신뢰를 얻습니다.")


if __name__ == "__main__":
    main()
