"""
qc.py — 결측을 메우기 전에 결측이 어떤 종류인지 먼저 봅니다.

대치 방법을 고르려면 결측 기전을 알아야 합니다. 세 가지가 섞여 있고 처리가
전부 다릅니다.

  무작위 결측       흩어져 있고 값의 크기와 무관.
                    -> 중앙값 대치로 충분합니다.

  검출한계 결측     낮게 발현되는 feature 가 결측. 결측률과 평균값이 음의 상관.
                    -> 중앙값으로 메우면 그 feature 를 **위로** 끌어올립니다.
                       실제로는 낮은 값이므로 좌측중도절단 대치가 맞습니다.

  표본/배치 실패    특정 표본이나 피험자에서 블록이 통째로 비어 있음.
                    -> 대치할 근거가 없습니다. 그런데도 중앙값으로 메우면 그
                       표본은 "가장 평균적인 표본"이 되어 군집 한가운데로
                       끌려갑니다. 진짜 신호가 아니라 대치가 만든 위치입니다.

`censoring_report` 는 앞의 두 가지를 구분해 주고,
`sample_missingness` / `flag_incomplete` 는 세 번째를 잡아냅니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mox.io import SAMPLE_COL, SUBJECT_COL


def censoring_report(X: np.ndarray) -> dict:
    """
    feature 별 결측률과 평균값의 관계를 봅니다.

    상관이 뚜렷한 음수면 검출한계 결측(MNAR)입니다. 낮은 값일수록 자주
    결측된다는 뜻이므로 중앙값 대치는 편향을 만듭니다.

    X: samples x features
    """
    X = np.asarray(X, dtype=float)
    na = np.isnan(X).mean(axis=0)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(X, axis=0)
    ok = np.isfinite(mu) & np.isfinite(na)
    if ok.sum() < 3 or np.std(na[ok]) < 1e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(na[ok], mu[ok])[0, 1])

    bands, edges = [], [(0, 1e-9), (1e-9, 0.05), (0.05, 0.20), (0.20, 1.01)]
    for lo, hi in edges:
        m = (na >= lo) & (na < hi) & ok
        bands.append({"na_from": lo, "na_to": hi, "n_features": int(m.sum()),
                      "mean_value": float(mu[m].mean()) if m.any() else float("nan")})
    return {"na_rate_overall": float(np.isnan(X).mean()),
            "corr_na_vs_mean": corr,
            "likely_left_censored": bool(np.isfinite(corr) and corr < -0.15),
            "bands": pd.DataFrame(bands)}


def sample_missingness(blocks: dict, md: pd.DataFrame) -> pd.DataFrame:
    """표본 x 블록 결측률. 블록이 통째로 빈 표본을 찾는 용도입니다."""
    out = pd.DataFrame({SAMPLE_COL: md[SAMPLE_COL].to_numpy(),
                        SUBJECT_COL: md[SUBJECT_COL].to_numpy()})
    for name, df in blocks.items():
        out[name] = np.isnan(df.to_numpy(dtype=float)).mean(axis=1)
    return out


def subject_missingness(blocks: dict, md: pd.DataFrame) -> pd.DataFrame:
    """
    피험자 단위 결측률.

    표본 단위로 보면 흩어져 보이던 결측이 피험자 단위로 모으면 몇 명에게
    몰려 있는 경우가 많습니다. 그건 무작위 결측이 아니라 그 사람의 검체나
    배치에 문제가 있었다는 뜻입니다.
    """
    s = sample_missingness(blocks, md)
    return s.drop(columns=[SAMPLE_COL]).groupby(SUBJECT_COL).mean().reset_index()


def flag_incomplete(blocks: dict, md: pd.DataFrame,
                    max_missing: float = 0.20) -> np.ndarray:
    """
    어느 한 블록에서라도 결측률이 max_missing 을 넘는 표본을 True 로 표시합니다.

    이 표본들을 어떻게 할지는 호출하는 쪽이 정합니다. 다만 조용히 대치해서
    분석에 넣는 것만은 피해야 합니다 — 그 표본의 위치는 자료가 아니라 대치가
    정한 것이 됩니다.
    """
    s = sample_missingness(blocks, md)
    cols = [c for c in s.columns if c not in (SAMPLE_COL, SUBJECT_COL)]
    return (s[cols].to_numpy() > max_missing).any(axis=1)


def report(blocks: dict, md: pd.DataFrame, max_missing: float = 0.20) -> str:
    """사람이 읽는 요약. 파이프라인 시작 전에 한 번 찍어 보라는 용도입니다."""
    lines = ["=" * 62, "결측 진단", "=" * 62]
    for name, df in blocks.items():
        X = df.to_numpy(dtype=float)
        rep = censoring_report(X)
        lines.append(f"\n[{name}] {X.shape[0]} 표본 x {X.shape[1]} feature  "
                     f"전체 결측 {rep['na_rate_overall']:.4f}")
        if rep["na_rate_overall"] == 0:
            lines.append("  결측 없음")
            continue
        lines.append(f"  결측률 vs 평균값 상관 {rep['corr_na_vs_mean']:+.3f}"
                     f"  -> {'검출한계 결측으로 보임' if rep['likely_left_censored'] else '무작위 결측에 가까움'}")
        for _, r in rep["bands"].iterrows():
            if r["n_features"]:
                lines.append(f"    결측률 {r['na_from']:.2f}-{r['na_to']:<4.2f}: "
                             f"feature {int(r['n_features']):5d}개, 평균 {r['mean_value']:7.2f}")

    subj = subject_missingness(blocks, md)
    cols = [c for c in subj.columns if c != SUBJECT_COL]
    worst = subj.loc[subj[cols].max(axis=1) > 1e-9].sort_values(
        by=cols, ascending=False)
    lines.append(f"\n[피험자 단위] 결측이 있는 피험자 {len(worst)} / {len(subj)}")
    for _, r in worst.head(10).iterrows():
        detail = "  ".join(f"{c} {r[c]:.3f}" for c in cols if r[c] > 1e-9)
        lines.append(f"    {r[SUBJECT_COL]}: {detail}")

    flag = flag_incomplete(blocks, md, max_missing)
    lines.append(f"\n[표본 제외 후보] 결측률 {max_missing:.0%} 초과 표본 "
                 f"{int(flag.sum())} / {len(md)}")
    if flag.any():
        lines.append("    " + ", ".join(md.loc[flag, SAMPLE_COL].astype(str)[:12]))
    lines.append("=" * 62)
    return "\n".join(lines)
