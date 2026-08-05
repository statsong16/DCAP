"""
cluster.py — 층위(stratum)를 만들고, 만든 것이 진짜인지 따집니다.

군집은 언제나 나옵니다. k 를 주면 KMeans 는 반드시 k 개를 돌려줍니다.
그래서 "나눴다"는 결과가 아니고, 다음 네 가지를 함께 보고서야 결과가 됩니다.

  1. k 선택        실루엣. 단조 증가하면 구조가 없다는 신호로 읽습니다.
  2. 안정성        **그룹(피험자)** 을 부트스트랩해 다시 군집하고 ARI 를 봅니다.
                   표본을 리샘플하면 같은 사람의 다른 방문이 복제본으로 들어가
                   안정성이 과대평가됩니다.
  3. 그룹 일관성   한 사람의 반복측정이 같은 군집에 들어가는 비율.
                   낮으면 군집이 사람이 아니라 측정 시점의 잡음을 잡고 있습니다.
  4. 메타데이터 연관  **그룹 단위**로 검정합니다. 표본 단위로 하면 한 사람을
                   여러 번 세는 셈이라 p 값이 실제보다 작게 나옵니다.
                   변수를 여럿 보므로 BH 보정도 함께 냅니다.

여기서 나오는 값은 전체 자료에 적합한 **기술 통계**입니다. held-out 추정치가
아닙니다. 일반화 성능은 evaluate.py + splits.py 로 따로 재야 합니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from mox.integrate import MultiOmicsEmbedder
from mox.io import SUBJECT_COL


def choose_k(Z: np.ndarray, k_range=range(2, 7), seed: int = 0):
    """실루엣으로 k 를 고릅니다. (best_k, 표) 를 돌려줍니다."""
    rows = []
    for k in k_range:
        lab = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(Z)
        rows.append({"k": int(k), "silhouette": float(silhouette_score(Z, lab))})
    tab = pd.DataFrame(rows)
    return int(tab.loc[tab["silhouette"].idxmax(), "k"]), tab


def fit_clusters(Z: np.ndarray, k: int, seed: int = 0):
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Z)
    return km, km.labels_


def group_consistency(md: pd.DataFrame, labels: np.ndarray,
                      group_col: str = SUBJECT_COL) -> float:
    """한 그룹의 모든 표본이 같은 군집에 들어간 비율."""
    t = md[[group_col]].copy()
    t["cluster"] = labels
    return float((t.groupby(group_col)["cluster"].nunique() == 1).mean())


def bootstrap_stability(blocks: dict, md: pd.DataFrame, k: int,
                        n_components: int = 10, seed: int = 0, n_boot: int = 20,
                        group_col: str = SUBJECT_COL, embedder_kwargs=None):
    """
    그룹 부트스트랩 안정성. (평균 ARI, 표준편차, 기준 라벨) 을 돌려줍니다.

    매 반복마다 전처리와 PCA 까지 다시 fit 합니다. 임베딩을 고정해 두고
    군집만 다시 하면 전처리의 불안정성이 통째로 숨습니다.
    """
    embedder_kwargs = embedder_kwargs or {}
    rng = np.random.default_rng(seed)
    groups = md[group_col].to_numpy()
    uniq = np.unique(groups)

    ref = MultiOmicsEmbedder(n_components, seed, **embedder_kwargs).fit(blocks)
    ref_lab = fit_clusters(ref.transform(blocks), k, seed)[1]

    scores = []
    for b in range(n_boot):
        drawn = np.unique(rng.choice(uniq, size=len(uniq), replace=True))
        idx = np.concatenate([np.where(groups == g)[0] for g in drawn])
        emb = MultiOmicsEmbedder(n_components, seed + b, **embedder_kwargs).fit(blocks, idx)
        lab = fit_clusters(emb.transform(blocks, idx), k, seed + b)[1]
        scores.append(adjusted_rand_score(ref_lab[idx], lab))
    return float(np.mean(scores)), float(np.std(scores)), ref_lab


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q 값."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    q = np.empty_like(p)
    q[order] = np.minimum.accumulate(
        (p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def association_tests(md: pd.DataFrame, labels: np.ndarray,
                      continuous=("Age", "BMI"),
                      categorical=("Gender", "enteroType"),
                      within_group=(), group_col: str = SUBJECT_COL) -> pd.DataFrame:
    """
    군집과 메타데이터의 연관.

    continuous / categorical 은 **그룹 단위**로 검정합니다 (그룹당 대표값 1개).
    within_group 은 그룹 안에서 변하는 변수(방문 차수 등)라 그룹 단위로 접을 수
    없으므로 표본 단위로 검정하고, 표에 그렇게 표시합니다.
    """
    t = md.copy()
    t["cluster"] = labels

    agg = {c: (c, "mean") for c in continuous if c in t.columns}
    agg.update({c: (c, lambda s: s.value_counts().idxmax())
                for c in categorical if c in t.columns})
    agg["cluster"] = ("cluster", lambda s: s.value_counts().idxmax())
    grouped = t.groupby(group_col).agg(**agg).reset_index()

    rows = []
    for col in continuous:
        if col not in grouped.columns:
            continue
        parts = [g[col].to_numpy() for _, g in grouped.groupby("cluster") if len(g) > 1]
        if len(parts) < 2:
            continue
        h, p = stats.kruskal(*parts)
        rows.append({"variable": col, "level": "group", "n": len(grouped),
                     "test": "Kruskal-Wallis", "statistic": float(h), "p": float(p)})
    for col in categorical:
        if col not in grouped.columns:
            continue
        ct = pd.crosstab(grouped["cluster"], grouped[col])
        if min(ct.shape) < 2:
            continue
        chi2, p, _, _ = stats.chi2_contingency(ct)
        rows.append({"variable": col, "level": "group", "n": len(grouped),
                     "test": "chi-square", "statistic": float(chi2), "p": float(p)})
    for col in within_group:
        if col not in t.columns:
            continue
        ct = pd.crosstab(labels, t[col])
        if min(ct.shape) < 2:
            continue
        chi2, p, _, _ = stats.chi2_contingency(ct)
        rows.append({"variable": col, "level": "sample (반복측정)", "n": len(t),
                     "test": "chi-square", "statistic": float(chi2), "p": float(p)})

    tab = pd.DataFrame(rows)
    if len(tab):
        tab["q_BH"] = _bh(tab["p"].to_numpy())
    return tab
