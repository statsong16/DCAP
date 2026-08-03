"""
stratify.py — 3개 오믹스를 합쳐 표본을 층위(stratum)로 나눕니다.

  블록별 전처리 -> 블록별 PCA 10개 -> 이어붙여 30차원 -> KMeans

k 는 실루엣으로 고르고, 고른 뒤에 세 가지를 같이 봅니다.

  1. 안정성   — 피험자를 부트스트랩해 다시 군집하고 원래 배정과의 ARI.
                표본이 아니라 피험자를 리샘플합니다. 표본을 리샘플하면 같은
                사람의 다른 방문이 복제본으로 들어가 안정성이 과대평가됩니다.
  2. 피험자 일관성 — 한 사람의 4회 방문이 같은 군집에 들어가는 비율.
                낮으면 군집이 사람이 아니라 그날의 상태를 잡고 있다는 뜻입니다.
  3. 메타데이터 연관 — 각 변수가 군집에 따라 다른가.
                검정은 피험자 단위(n=69)로 합니다. 표본 단위(n=276)로 하면
                한 사람을 네 번 세는 셈이라 p 값이 실제보다 작게 나옵니다.

여기서 나오는 수치는 전체 자료에 적합한 **기술 통계**입니다. held-out 추정치가
아닙니다. 일반화 성능은 predict.py 가 피험자 단위 GroupKFold 로 따로 잽니다.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from data import MultiOmicsEmbedder, load_blocks

CONTINUOUS = ("Age", "BMI")
CATEGORICAL = ("Gender", "enteroType", "subtype")


def choose_k(Z: np.ndarray, k_range, seed: int):
    rows = []
    for k in k_range:
        lab = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(Z)
        rows.append({"k": k, "silhouette": float(silhouette_score(Z, lab))})
    tab = pd.DataFrame(rows)
    best = int(tab.loc[tab["silhouette"].idxmax(), "k"])
    return best, tab


def bootstrap_stability(blocks, md, k, n_comp, seed, n_boot):
    """피험자 단위 부트스트랩. 겹치는 표본에서만 ARI 를 계산합니다."""
    rng = np.random.default_rng(seed)
    subjects = md["metadata.ID"].to_numpy()
    uniq = np.unique(subjects)
    ref_emb = MultiOmicsEmbedder(n_comp, seed).fit(blocks, np.arange(len(md)))
    ref_lab = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(
        ref_emb.transform(blocks))

    scores = []
    for b in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(subjects == s)[0] for s in np.unique(draw)])
        emb = MultiOmicsEmbedder(n_comp, seed + b).fit(blocks, idx)
        Z = emb.transform(blocks, idx)
        lab = KMeans(n_clusters=k, n_init=20, random_state=seed + b).fit_predict(Z)
        scores.append(adjusted_rand_score(ref_lab[idx], lab))
    return float(np.mean(scores)), float(np.std(scores)), ref_lab


def subject_level_table(md: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """피험자마다 최빈 군집과 피험자 상수 변수를 모읍니다."""
    t = md.copy()
    t["cluster"] = labels
    agg = t.groupby("metadata.ID").agg(
        cluster=("cluster", lambda s: s.value_counts().idxmax()),
        Age=("Age", "mean"), BMI=("BMI", "mean"),
        Gender=("Gender", "first"),
        enteroType=("enteroType", lambda s: s.value_counts().idxmax()),
    ).reset_index()
    return agg


def association_tests(md: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    subj = subject_level_table(md, labels)
    rows = []
    for col in CONTINUOUS:
        groups = [g[col].to_numpy() for _, g in subj.groupby("cluster") if len(g) > 1]
        if len(groups) < 2:
            continue
        h, p = stats.kruskal(*groups)
        rows.append({"variable": col, "level": "subject", "n": len(subj),
                     "test": "Kruskal-Wallis", "statistic": float(h), "p": float(p)})
    for col in ("Gender", "enteroType"):
        ct = pd.crosstab(subj["cluster"], subj[col])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2, p, _, _ = stats.chi2_contingency(ct)
        rows.append({"variable": col, "level": "subject", "n": len(subj),
                     "test": "chi-square", "statistic": float(chi2), "p": float(p)})
    # subtype(방문)은 피험자 상수가 아니므로 표본 단위로만 볼 수 있습니다.
    ct = pd.crosstab(labels, md["subtype"])
    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
        chi2, p, _, _ = stats.chi2_contingency(ct)
        rows.append({"variable": "subtype", "level": "sample (반복측정 있음)",
                     "n": len(md), "test": "chi-square",
                     "statistic": float(chi2), "p": float(p)})
    tab = pd.DataFrame(rows)
    # 변수 5개를 한 번에 검정하므로 보정 없이 읽으면 안 됩니다 (Benjamini-Hochberg).
    p = tab["p"].to_numpy()
    order = np.argsort(p)
    q = np.empty_like(p)
    q[order] = np.minimum.accumulate(
        (p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    tab["q_BH"] = np.clip(q, 0, 1)
    return tab


def main(data_dir: str, out_dir: str, k_min: int, k_max: int,
         n_comp: int, seed: int, n_boot: int):
    os.makedirs(out_dir, exist_ok=True)
    blocks, md = load_blocks(data_dir)
    all_idx = np.arange(len(md))

    emb = MultiOmicsEmbedder(n_comp, seed).fit(blocks, all_idx)
    Z = emb.transform(blocks)
    print(f"[embed] {Z.shape[1]}차원 (블록당 {n_comp}개) · 표본 {Z.shape[0]}개 · "
          f"피험자 {md['metadata.ID'].nunique()}명")
    for b, v in emb.var_explained_.items():
        print(f"[embed]   {b:11s} PC1-{n_comp} 설명 분산 {100 * v:5.1f}%")

    k, ktab = choose_k(Z, range(k_min, k_max + 1), seed)
    print("\n[k 선택] 실루엣")
    print(ktab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"[k 선택] k = {k}")

    ari_mean, ari_sd, labels = bootstrap_stability(blocks, md, k, n_comp, seed, n_boot)
    sil = float(silhouette_score(Z, labels))

    tmp = md.copy()
    tmp["cluster"] = labels
    consistency = float((tmp.groupby("metadata.ID")["cluster"].nunique() == 1).mean())

    print(f"\n[군집] 크기 {np.bincount(labels, minlength=k).tolist()}")
    print(f"[군집] 실루엣 {sil:.3f}")
    print(f"[군집] 부트스트랩 ARI {ari_mean:.3f} ± {ari_sd:.3f}  (피험자 리샘플 {n_boot}회)")
    print(f"[군집] 피험자 일관성 {consistency:.3f}  "
          f"(4회 방문이 모두 같은 군집인 피험자 비율)")

    assoc = association_tests(md, labels)
    print("\n[메타데이터 연관] 검정은 피험자 단위(n=69). 표본 단위는 반복측정 때문에 p 가 부풀려집니다.")
    print(assoc.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    truth_path = os.path.join(data_dir, "truth_stratum.csv")
    ari_truth, bmi_ceiling = None, None
    if os.path.exists(truth_path):
        truth = pd.read_csv(truth_path)["true_stratum"].to_numpy()
        ari_truth = float(adjusted_rand_score(truth, labels))
        # 정답 층위를 그대로 알려줬을 때의 BMI R². predict.py 결과를 읽을 때
        # 기준이 되는 상한입니다 (합성 자료에서만 계산 가능).
        y = md["BMI"].to_numpy()
        fitted = np.array([y[truth == g].mean() for g in np.unique(truth)])[truth]
        bmi_ceiling = float(1 - ((y - fitted) ** 2).mean() / y.var())
        print(f"\n[합성 자료 채점] 정답 층위와의 ARI {ari_truth:.3f}")
        print(f"[합성 자료 채점] 정답 층위로 설명되는 BMI 분산 {bmi_ceiling:.3f} "
              f"(= predict.py BMI R² 의 상한)")

    pd.DataFrame({"sample.ID": md["sample.ID"], "metadata.ID": md["metadata.ID"],
                  "cluster": labels}).to_csv(
        os.path.join(out_dir, "clusters.csv"), index=False)
    ktab.to_csv(os.path.join(out_dir, "k_selection.csv"), index=False)
    assoc.to_csv(os.path.join(out_dir, "association.csv"), index=False)
    np.savez(os.path.join(out_dir, "embedding.npz"), Z=Z,
             sample_id=md["sample.ID"].to_numpy())
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"seed": seed, "k": k, "n_components_per_block": n_comp,
                   "silhouette": sil, "bootstrap_ari_mean": ari_mean,
                   "bootstrap_ari_sd": ari_sd, "n_boot": n_boot,
                   "subject_consistency": consistency,
                   "cluster_sizes": np.bincount(labels, minlength=k).tolist(),
                   "ari_vs_truth": ari_truth, "bmi_r2_ceiling": bmi_ceiling,
                   "var_explained": emb.var_explained_}, f, indent=2)
    print(f"\n[saved] {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/stratify")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=6)
    ap.add_argument("--n-components", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.data, a.out, a.k_min, a.k_max, a.n_components, a.seed, a.n_boot)
