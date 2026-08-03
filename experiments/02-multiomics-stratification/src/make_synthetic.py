"""
make_synthetic.py — multi_omics_set.RData 없이도 파이프라인이 도는지 확인하는 용도.

export_real.py 가 만드는 것과 같은 형식의 CSV 4개를 생성합니다.
실제 자료의 구조를 일부러 흉내냈습니다:

  - 피험자 69명 x 방문 4회 = 표본 276개 (반복측정)
  - 피험자 고유 효과가 3개 블록 모두에 들어감  -> 표본 단위 분할은 새게 되어 있음
  - 잠재 층위(stratum) 3개가 피험자 단위로 배정되고 BMI 를 이동시킴
  - microbiome 은 희소한 조성 자료, metabolome 은 로그정규, proteome 은 NPX 유사

실제 분석에는 쓰지 마세요. 파이프라인 점검 전용입니다.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def main(out_dir: str, n_subjects: int = 69, n_visits: int = 4,
         n_msp: int = 1350, n_metab: int = 413, n_prot: int = 770,
         n_strata: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    # ---- 피험자 수준 변수 -------------------------------------------------
    stratum = rng.choice(n_strata, size=n_subjects, p=[0.5, 0.3, 0.2])
    gender = rng.choice(["Female", "Male"], size=n_subjects, p=[0.57, 0.43])
    age0 = rng.normal(57.9, 4.0, size=n_subjects).clip(51, 66)
    # 층위가 BMI 를 이동시킵니다. 층위 -> 메타데이터 신호는 여기서만 들어갑니다.
    bmi0 = 24.0 + np.array([0.0, 2.0, 4.0])[stratum] + rng.normal(0, 2.2, size=n_subjects)

    # 각 블록에서 층위가 건드리는 변수 (전체가 아니라 일부만)
    msp_hit = rng.random(n_msp) < 0.06
    met_hit = rng.random(n_metab) < 0.10
    pro_hit = rng.random(n_prot) < 0.08
    # 효과 크기는 일부러 작게 잡았습니다. 층위가 완벽히 복원되는 합성 자료는
    # 파이프라인이 제대로 도는지 구별해 주지 못합니다.
    msp_eff = rng.normal(0, 0.85, size=(n_strata, n_msp)) * msp_hit
    met_eff = rng.normal(0, 0.68, size=(n_strata, n_metab)) * met_hit
    pro_eff = rng.normal(0, 0.47, size=(n_strata, n_prot)) * pro_hit

    # 블록별 기준선
    msp_base = rng.normal(-2.0, 1.6, size=n_msp)
    met_base = rng.normal(13.0, 2.2, size=n_metab)       # 자연로그 스케일
    pro_base = rng.normal(3.0, 2.0, size=n_prot)         # NPX 스케일

    # 피험자 고유 효과 — 같은 사람의 4회 방문이 서로 닮게 만드는 항
    msp_sub = rng.normal(0, 1.0, size=(n_subjects, n_msp))
    met_sub = rng.normal(0, 0.8, size=(n_subjects, n_metab))
    pro_sub = rng.normal(0, 0.6, size=(n_subjects, n_prot))

    sample_ids, meta_rows = [], []
    MIC, MET, PRO = [], [], []

    for i in range(n_subjects):
        subj = f"S{2800 + i * 7:04d}"
        s = stratum[i]
        for v in range(1, n_visits + 1):
            sid = f"{subj}_v{v}"
            sample_ids.append(sid)

            # --- microbiome: 희소 조성 자료 -------------------------------
            log_ab = msp_base + msp_sub[i] + msp_eff[s] + rng.normal(0, 0.6, size=n_msp)
            ab = np.exp(log_ab)
            ab *= (rng.random(n_msp) > 0.84)             # 실제 자료의 0 비율 ~0.84
            if ab.sum() == 0:
                ab[rng.integers(0, n_msp)] = 1.0
            # 실제 자료는 열 합이 1이 아니라 3e-4 근처에서 흔들립니다.
            ab = ab / ab.sum() * rng.normal(3.2e-4, 0.4e-4)
            MIC.append(ab)

            # --- metabolome: 로그정규 강도 --------------------------------
            lm = met_base + met_sub[i] + met_eff[s] + rng.normal(0, 0.7, size=n_metab)
            mv = np.exp(lm)
            mv[rng.random(n_metab) < 0.009] = 0.0        # 검출 한계 아래
            mv[rng.random(n_metab) < 0.014] = np.nan     # 결측
            MET.append(mv)

            # --- proteome: NPX 유사 (이미 log2) ---------------------------
            pv = pro_base + pro_sub[i] + pro_eff[s] + rng.normal(0, 0.35, size=n_prot)
            pv[rng.random(n_prot) < 0.010] = np.nan
            PRO.append(pv)

            meta_rows.append({
                "sample.ID": sid,
                "metadata.ID": subj,
                "type": "Case",
                "subtype": f"v{v}",
                "Age": age0[i] + 0.33 * (v - 1),
                "Gender": gender[i],
                "BMI": bmi0[i] + rng.normal(0, 0.3),
                "Geography": "Sweden",
                "Sequencer": "HiSeq",
                # 실제 자료처럼 심하게 치우친 3범주. microbiome 에서 파생시킵니다.
                "enteroType": None,
            })

    MIC = np.asarray(MIC)                                 # (samples, features)
    MET = np.asarray(MET)
    PRO = np.asarray(PRO)

    # enterotype 은 실제 자료에서도 microbiome 으로 정의된 값이므로 여기서도 파생시킵니다.
    # 지배 MSP 3개 중 어느 것이 가장 큰가로 배정하고, 실제와 같이 심하게 치우치게 둡니다.
    dom = np.argsort(-MIC.sum(axis=0))[:3]
    share = MIC[:, dom] / (MIC[:, dom].sum(axis=1, keepdims=True) + 1e-30)
    et_names = np.array(["ET-Firmicutes", "ET-Prevotella", "ET-Bacteroides"])
    et = np.where(share[:, 1] > 0.62, 1, np.where(share[:, 2] > 0.70, 2, 0))
    for row, e in zip(meta_rows, et):
        row["enteroType"] = et_names[e]

    md = pd.DataFrame(meta_rows)
    msp_names = [f"msp_{i + 1:04d}" for i in range(n_msp)]
    met_names = [f"Metabolite_{i + 1:03d}" for i in range(n_metab)]
    pro_names = [f"PROT{i + 1:03d}" for i in range(n_prot)]

    # 실제 RData 와 같은 방향으로 저장합니다: 행 = feature, 열 = sample
    pd.DataFrame(MIC.T, index=msp_names, columns=sample_ids).to_csv(
        os.path.join(out_dir, "microbiome.csv"))
    pd.DataFrame(MET.T, index=met_names, columns=sample_ids).to_csv(
        os.path.join(out_dir, "metabolome.csv"))
    pd.DataFrame(PRO.T, index=pro_names, columns=sample_ids).to_csv(
        os.path.join(out_dir, "proteome.csv"))
    md.to_csv(os.path.join(out_dir, "metadata.csv"), index=False)

    # 정답 층위는 합성 자료에서만 존재합니다. 군집이 이것을 되찾는지 채점하는 용도.
    pd.DataFrame({"metadata.ID": md["metadata.ID"],
                  "true_stratum": np.repeat(stratum, n_visits)}).to_csv(
        os.path.join(out_dir, "truth_stratum.csv"), index=False)

    print(f"[synthetic] {out_dir}  samples={len(sample_ids)} subjects={n_subjects} "
          f"visits={n_visits} seed={seed}")
    print(f"[synthetic] microbiome {MIC.shape[1]}x{MIC.shape[0]}  "
          f"zeros={np.mean(MIC == 0):.3f}")
    print(f"[synthetic] metabolome {MET.shape[1]}x{MET.shape[0]}  "
          f"NA={np.mean(np.isnan(MET)):.3f}")
    print(f"[synthetic] proteome   {PRO.shape[1]}x{PRO.shape[0]}  "
          f"NA={np.mean(np.isnan(PRO)):.3f}")
    print(f"[synthetic] enterotype {md['enteroType'].value_counts().to_dict()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--subjects", type=int, default=69)
    ap.add_argument("--visits", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.out, n_subjects=a.subjects, n_visits=a.visits, seed=a.seed)
