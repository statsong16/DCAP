"""
run_describe.py — 코호트 기술 통계와 그림.

모델링 전에 자료를 먼저 봅니다. 여기서 잡히는 것들:

  - 상수 변수 (모든 표본이 같은 값) — 표적으로도 공변량으로도 쓸 수 없습니다.
  - 반복측정 구조 — 피험자당 표본 수, 피험자 내 변동.
    피험자 내 변동이 거의 없는 변수는 사실상 피험자 상수이고, 그런 표적에
    표본 단위 분할을 쓰면 성능이 재식별 정확도가 됩니다.
  - 블록별 결측 기전 (qc.report)

R 로 같은 것을 하려면 R/descriptive.R 을 보세요.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mox import qc  # noqa: E402
from mox.integrate import MultiOmicsEmbedder  # noqa: E402
from mox.io import SUBJECT_COL, load_blocks  # noqa: E402
from mox.splits import describe_repeated_measures  # noqa: E402

ET_COLORS = {"ET-Bacteroides": "#E15759", "ET-Firmicutes": "#4E79A7",
             "ET-Prevotella": "#59A14F"}

# 그림 안의 글자는 영어로 씁니다. 한글 폰트가 없는 환경에서 matplotlib 이
# 네모(tofu)로 렌더링하기 때문입니다. 콘솔 출력과 문서는 한글 그대로입니다.


def cohort_summary(md: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in md.columns:
        s = md[col]
        n_uniq = s.astype(str).nunique()
        row = {"variable": col, "n_unique": n_uniq,
               "constant": n_uniq == 1, "n_missing": int(s.isna().sum())}
        if pd.api.types.is_numeric_dtype(s):
            row.update({"mean": float(s.mean()), "sd": float(s.std()),
                        "min": float(s.min()), "max": float(s.max())})
        rows.append(row)
    return pd.DataFrame(rows)


def figure_cohort(md: pd.DataFrame, path: str):
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
    ax[0, 0].hist(md["Age"], bins=25, color="#4E79A7", edgecolor="white")
    ax[0, 0].set(title="A. Age distribution", xlabel="Age", ylabel="Count")

    groups = [g["BMI"].to_numpy() for _, g in md.groupby("enteroType")]
    names = [n for n, _ in md.groupby("enteroType")]
    bp = ax[0, 1].boxplot(groups, tick_labels=names, patch_artist=True)
    for patch, n in zip(bp["boxes"], names):
        patch.set_facecolor(ET_COLORS.get(n, "#BAB0AC"))
    ax[0, 1].set(title="B. BMI by enterotype", ylabel="BMI")
    ax[0, 1].tick_params(axis="x", rotation=15)

    g = md["Gender"].value_counts()
    ax[1, 0].bar(g.index, g.to_numpy(), color=["#F28E2B", "#76B7B2"], width=0.6)
    for i, v in enumerate(g.to_numpy()):
        ax[1, 0].text(i, v, str(v), ha="center", va="bottom")
    ax[1, 0].set(title="C. Sex composition", ylabel="Count")

    e = md["enteroType"].value_counts()
    ax[1, 1].bar(e.index, e.to_numpy(),
                 color=[ET_COLORS.get(n, "#BAB0AC") for n in e.index], width=0.7)
    for i, v in enumerate(e.to_numpy()):
        ax[1, 1].text(i, v, str(v), ha="center", va="bottom")
    ax[1, 1].set(title="D. Enterotype composition", ylabel="Count")
    ax[1, 1].tick_params(axis="x", rotation=15)

    fig.suptitle(f"Cohort description (n={len(md)} samples, {md[SUBJECT_COL].nunique()} subjects)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_pca(blocks: dict, md: pd.DataFrame, block: str, path: str, seed: int):
    """
    한 블록의 PCA 산점도.

    기술 통계이므로 전체 자료에 적합합니다. 예측 성능을 재는 것이 아니라
    자료를 눈으로 보는 단계입니다 — 이 그림의 분리를 성능으로 읽으면 안 됩니다.
    """
    emb = MultiOmicsEmbedder(2, seed).fit({block: blocks[block]})
    Z = emb.transform({block: blocks[block]})
    ve = emb.pcas_[block].explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for name, sub in md.assign(PC1=Z[:, 0], PC2=Z[:, 1]).groupby("enteroType"):
        ax.scatter(sub["PC1"], sub["PC2"], s=16, alpha=0.85, label=name,
                   color=ET_COLORS.get(name, "#BAB0AC"))
    ax.set(title=f"{block} PCA",
           xlabel=f"PC1 ({100 * ve[0]:.1f}% variance)",
           ylabel=f"PC2 ({100 * ve[1]:.1f}% variance)")
    ax.legend(title="Enterotype", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return ve


def main(data_dir: str, out_dir: str, seed: int):
    os.makedirs(out_dir, exist_ok=True)
    blocks, md = load_blocks(data_dir)

    print(f"표본 {len(md)} · 피험자 {md[SUBJECT_COL].nunique()}\n")
    summary = cohort_summary(md)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    const = summary.loc[summary["constant"], "variable"].tolist()
    if const:
        print(f"\n[주의] 상수 변수 {const} — 표적으로도 공변량으로도 쓸 수 없습니다")

    rm = describe_repeated_measures(
        md, constant_candidates=("Age", "BMI", "Gender", "enteroType"))
    print("\n반복측정 구조:")
    print(rm.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("  -> 피험자 내 변동이 0에 가까운 변수는 사실상 피험자 상수입니다.")
    print("     그런 표적에 표본 단위 분할을 쓰면 성능은 재식별 정확도가 됩니다.")

    print()
    print(qc.report(blocks, md))

    figure_cohort(md, os.path.join(out_dir, "cohort.png"))
    ve = figure_pca(blocks, md, "proteome", os.path.join(out_dir, "pca_proteome.png"), seed)
    summary.to_csv(os.path.join(out_dir, "cohort_summary.csv"), index=False)
    rm.to_csv(os.path.join(out_dir, "repeated_measures.csv"), index=False)
    print(f"\n[saved] {out_dir}/cohort.png, pca_proteome.png "
          f"(PC1 {100 * ve[0]:.1f}%, PC2 {100 * ve[1]:.1f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/describe")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.data, a.out, a.seed)
