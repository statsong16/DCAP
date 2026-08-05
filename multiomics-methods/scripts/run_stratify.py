"""
run_stratify.py — 세 블록을 합쳐 표본 층위를 만들고, 그 층위가 진짜인지 따집니다.

여기서 나오는 값은 전체 자료에 적합한 **기술 통계**입니다. held-out 추정치가
아닙니다. 층위를 예측에 쓸 수 있는지는 run_predict.py 가 따로 잽니다.

--drop-incomplete 를 주면 블록 결측률이 큰 표본을 먼저 뺍니다. 왜 그래야
하는지는 run_missing_data.py 가 숫자로 보여 줍니다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mox import qc  # noqa: E402
from mox.cluster import (association_tests, bootstrap_stability, choose_k,  # noqa: E402
                         group_consistency)
from mox.integrate import MultiOmicsEmbedder  # noqa: E402
from mox.io import SAMPLE_COL, SUBJECT_COL, load_blocks  # noqa: E402


def main(data_dir: str, out_dir: str, k_min: int, k_max: int, n_comp: int,
         seed: int, n_boot: int, drop_incomplete: float | None):
    os.makedirs(out_dir, exist_ok=True)
    blocks, md = load_blocks(data_dir)

    if drop_incomplete is not None:
        flag = qc.flag_incomplete(blocks, md, drop_incomplete)
        if flag.any():
            print(f"[제외] 블록 결측률 {drop_incomplete:.0%} 초과 표본 {int(flag.sum())}개 "
                  f"(피험자 {md.loc[flag, SUBJECT_COL].nunique()}명)")
            blocks = {k: v.loc[~flag].reset_index(drop=True) for k, v in blocks.items()}
            md = md.loc[~flag].reset_index(drop=True)

    emb = MultiOmicsEmbedder(n_comp, seed).fit(blocks)
    Z = emb.transform(blocks)
    print(emb.summary())
    print(f"표본 {len(md)} · 피험자 {md[SUBJECT_COL].nunique()}")

    k, ktab = choose_k(Z, range(k_min, k_max + 1), seed)
    print("\n[k 선택] 실루엣")
    print(ktab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if k == max(ktab["k"]):
        print("  [주의] 실루엣이 끝까지 오르고 있습니다. 뚜렷한 군집 구조가 "
              "없다는 신호일 수 있습니다.")
    print(f"[k 선택] k = {k}")

    ari_mean, ari_sd, labels = bootstrap_stability(blocks, md, k, n_comp, seed, n_boot)
    sil = float(silhouette_score(Z, labels))
    cons = group_consistency(md, labels)
    print(f"\n[군집] 크기 {np.bincount(labels, minlength=k).tolist()}")
    print(f"[군집] 실루엣 {sil:.3f}")
    print(f"[군집] 부트스트랩 ARI {ari_mean:.3f} ± {ari_sd:.3f} (피험자 리샘플 {n_boot}회)")
    print(f"[군집] 피험자 일관성 {cons:.3f} (반복측정이 모두 같은 군집인 피험자 비율)")

    assoc = association_tests(md, labels, within_group=("subtype",))
    print("\n[메타데이터 연관] 검정은 피험자 단위. 표본 단위로 하면 한 사람을 "
          "여러 번 세어 p 가 부풀려집니다.")
    print(assoc.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    if len(assoc) and (assoc["q_BH"] >= 0.05).all():
        print("  -> BH 보정 후 유의한 변수 없음.")

    # 합성 자료에만 있는 정답. 표본을 제외했을 수 있으므로 sample.ID 로 맞춥니다.
    ari_truth = bmi_ceiling = None
    truth_path = os.path.join(data_dir, "truth_stratum.csv")
    if os.path.exists(truth_path):
        truth = pd.read_csv(truth_path).set_index(SAMPLE_COL)["true_stratum"]
        aligned = truth.reindex(md[SAMPLE_COL].astype(str))
        if aligned.notna().all():
            ari_truth = float(adjusted_rand_score(aligned.to_numpy(), labels))
            print(f"\n[합성 자료 채점] 정답 층위와의 ARI {ari_truth:.3f}")
            # 정답 층위를 그대로 알려줬을 때의 BMI R². run_predict.py 의 BMI 결과를
            # 읽을 때 기준이 되는 상한입니다.
            g = aligned.to_numpy()
            y = md["BMI"].to_numpy()
            fitted = np.array([y[g == v].mean() for v in np.unique(g)])[
                np.searchsorted(np.unique(g), g)]
            bmi_ceiling = float(1 - ((y - fitted) ** 2).mean() / y.var())
            print(f"[합성 자료 채점] 정답 층위로 설명되는 BMI 분산 {bmi_ceiling:.3f} "
                  f"(= BMI R² 의 상한)")

    pd.DataFrame({SAMPLE_COL: md[SAMPLE_COL], SUBJECT_COL: md[SUBJECT_COL],
                  "cluster": labels}).to_csv(
        os.path.join(out_dir, "clusters.csv"), index=False)
    ktab.to_csv(os.path.join(out_dir, "k_selection.csv"), index=False)
    assoc.to_csv(os.path.join(out_dir, "association.csv"), index=False)
    np.savez(os.path.join(out_dir, "embedding.npz"), Z=Z,
             sample_id=md[SAMPLE_COL].to_numpy())
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"seed": seed, "k": k, "n_components_per_block": n_comp,
                   "n_samples": len(md), "n_subjects": int(md[SUBJECT_COL].nunique()),
                   "drop_incomplete": drop_incomplete,
                   "silhouette": sil, "bootstrap_ari_mean": ari_mean,
                   "bootstrap_ari_sd": ari_sd, "n_boot": n_boot,
                   "group_consistency": cons, "ari_vs_truth": ari_truth,
                   "bmi_r2_ceiling": bmi_ceiling,
                   "cluster_sizes": np.bincount(labels, minlength=k).tolist(),
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
    ap.add_argument("--drop-incomplete", type=float, default=None,
                    help="블록 결측률이 이 값을 넘는 표본을 제외 (예: 0.2)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.data, a.out, a.k_min, a.k_max, a.n_components, a.seed, a.n_boot,
         a.drop_incomplete)
