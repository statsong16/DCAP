"""
make_synthetic.py — 실제 자료 없이도 전체 파이프라인이 도는지 확인하는 용도.

이 저장소의 조건은 "자료 접근 없이 클론 하나로 재현된다" 입니다. 그래서
합성 자료가 장식이 아니라 검증 대상입니다. 실제 코호트 자료에서 관찰한
구조와 **병리**를 일부러 재현합니다.

  구조
    피험자 69명 x 방문 4회 = 표본 276개 (반복측정)
    피험자 고유 효과가 3개 블록 전부에 들어감 -> 표본 단위 분할은 새게 되어 있음
    잠재 층위 3개가 피험자 단위로 배정되고 BMI 를 이동시킴

  병리 1: 검출한계 결측 (proteome)
    전역 LOD 아래 값이 결측이 됩니다. 낮게 발현되는 단백질일수록 결측률이
    높아지므로 결측률과 평균값이 음의 상관을 갖습니다. 중앙값 대치는 이걸
    위로 끌어올립니다.

  병리 2: 피험자 단위 블록 결측 (metabolome)
    피험자 2명이 4회 방문 내내 같은 대사체 집합이 통째로 비어 있습니다.
    검체나 배치 실패로 생기는 모양이고, 중앙값으로 메우면 그 피험자가
    "가장 평균적인 사람"이 되어 군집 한가운데로 끌려갑니다.

실제 분석에 쓰지 마세요. 파이프라인 점검 전용입니다.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mox.io import write_blocks  # noqa: E402


def main(out_dir: str, n_subjects: int = 69, n_visits: int = 4,
         n_msp: int = 1350, n_metab: int = 413, n_prot: int = 770,
         n_strata: int = 3, effect_scale: float = 1.2, seed: int = 0):
    rng = np.random.default_rng(seed)

    # ---- 피험자 수준 --------------------------------------------------
    stratum = rng.choice(n_strata, size=n_subjects, p=[0.5, 0.3, 0.2])
    gender = rng.choice(["Female", "Male"], size=n_subjects, p=[0.57, 0.43])
    age0 = rng.normal(57.9, 4.0, size=n_subjects).clip(51, 66)
    # 층위 -> 메타데이터 신호는 BMI 하나에만 넣습니다. 나머지 표적이 우연 수준으로
    # 나오는 것이 정답이고, 그래야 파이프라인이 헛것을 잡는지 알 수 있습니다.
    bmi0 = 24.0 + np.array([0.0, 2.0, 4.0])[stratum] + rng.normal(0, 2.2, size=n_subjects)

    # 층위가 건드리는 feature 는 일부뿐입니다.
    # effect_scale 로 신호 세기를 조절합니다. 진단이 신호 있음/없음을 실제로
    # 구분하는지 보려면 두 값에서 돌려 보세요 (README 의 표가 그렇게 만들어졌습니다).
    e = effect_scale
    msp_eff = rng.normal(0, 0.85 * e, size=(n_strata, n_msp)) * (rng.random(n_msp) < 0.06)
    met_eff = rng.normal(0, 0.68 * e, size=(n_strata, n_metab)) * (rng.random(n_metab) < 0.10)
    pro_eff = rng.normal(0, 0.47 * e, size=(n_strata, n_prot)) * (rng.random(n_prot) < 0.08)

    msp_base = rng.normal(-2.0, 1.6, size=n_msp)
    met_base = rng.normal(13.0, 2.2, size=n_metab)      # 자연로그 스케일
    pro_base = rng.normal(3.0, 2.0, size=n_prot)        # NPX 스케일

    # 피험자 고유 효과 — 같은 사람의 방문들이 서로 닮게 만드는 항
    msp_sub = rng.normal(0, 1.0, size=(n_subjects, n_msp))
    met_sub = rng.normal(0, 0.8, size=(n_subjects, n_metab))
    pro_sub = rng.normal(0, 0.6, size=(n_subjects, n_prot))

    sample_ids, subject_ids, meta_rows = [], [], []
    MIC, MET, PRO = [], [], []
    for i in range(n_subjects):
        subj = f"S{2800 + i * 7:04d}"
        s = stratum[i]
        for v in range(1, n_visits + 1):
            sample_ids.append(f"{subj}_v{v}")
            subject_ids.append(subj)

            log_ab = msp_base + msp_sub[i] + msp_eff[s] + rng.normal(0, 0.6, size=n_msp)
            ab = np.exp(log_ab) * (rng.random(n_msp) > 0.84)
            if ab.sum() == 0:
                ab[rng.integers(0, n_msp)] = 1.0
            # 실제 자료처럼 표본별 총합이 흔들리게 둡니다 (시퀀싱 깊이 차이).
            MIC.append(ab / ab.sum() * rng.normal(3.2e-4, 0.4e-4))

            mv = np.exp(met_base + met_sub[i] + met_eff[s] + rng.normal(0, 0.7, size=n_metab))
            mv[rng.random(n_metab) < 0.009] = 0.0       # 검출 한계 아래 -> 0 으로 기록
            MET.append(mv)

            PRO.append(pro_base + pro_sub[i] + pro_eff[s] + rng.normal(0, 0.35, size=n_prot))

            meta_rows.append({
                "sample.ID": f"{subj}_v{v}", "metadata.ID": subj,
                "type": "Case", "subtype": f"v{v}",
                "Age": age0[i] + 0.33 * (v - 1), "Gender": gender[i],
                "BMI": bmi0[i] + rng.normal(0, 0.3),
                "Geography": "Sweden", "Sequencer": "HiSeq", "enteroType": None,
            })

    MIC, MET, PRO = np.asarray(MIC), np.asarray(MET), np.asarray(PRO)
    subject_ids = np.asarray(subject_ids)

    # ---- 병리 1: 검출한계 결측 (proteome) ------------------------------
    # 전역 LOD 아래를 결측 처리합니다. 낮게 발현되는 단백질일수록 많이 잘립니다.
    lod = -1.50
    PRO = np.where(PRO < lod, np.nan, PRO)

    # ---- 병리 2: 피험자 단위 블록 결측 (metabolome) ---------------------
    # 피험자 2명, 4회 방문 전부, 같은 대사체 집합이 비어 있습니다.
    failed = rng.choice(np.unique(subject_ids), size=2, replace=False)
    failure_frac = {failed[0]: 0.27, failed[1]: 0.73}
    for subj, frac in failure_frac.items():
        cols = rng.choice(n_metab, size=int(round(frac * n_metab)), replace=False)
        MET[np.ix_(subject_ids == subj, cols)] = np.nan

    # ---- enterotype: 실제 자료처럼 microbiome 에서 파생 ------------------
    dom = np.argsort(-MIC.sum(axis=0))[:3]
    share = MIC[:, dom] / (MIC[:, dom].sum(axis=1, keepdims=True) + 1e-30)
    names = np.array(["ET-Firmicutes", "ET-Prevotella", "ET-Bacteroides"])
    et = np.where(share[:, 1] > 0.62, 1, np.where(share[:, 2] > 0.70, 2, 0))
    for row, e in zip(meta_rows, et):
        row["enteroType"] = names[e]

    md = pd.DataFrame(meta_rows)
    blocks = {
        "microbiome": pd.DataFrame(MIC, index=sample_ids,
                                   columns=[f"msp_{i + 1:04d}" for i in range(n_msp)]),
        "metabolome": pd.DataFrame(MET, index=sample_ids,
                                   columns=[f"Metabolite_{i + 1:03d}" for i in range(n_metab)]),
        "proteome": pd.DataFrame(PRO, index=sample_ids,
                                 columns=[f"PROT{i + 1:03d}" for i in range(n_prot)]),
    }
    write_blocks(blocks, md, out_dir)
    pd.DataFrame({"sample.ID": md["sample.ID"], "metadata.ID": md["metadata.ID"],
                  "true_stratum": np.repeat(stratum, n_visits)}).to_csv(
        os.path.join(out_dir, "truth_stratum.csv"), index=False)
    # 검출한계 대치를 실제 LOD 로 해 보고 싶을 때 쓰는 참값입니다.
    pd.DataFrame({"feature": blocks["proteome"].columns,
                  "lod": np.full(n_prot, lod)}).to_csv(
        os.path.join(out_dir, "proteome_lod.csv"), index=False)

    print(f"[synthetic] {out_dir}  표본 {len(sample_ids)} · 피험자 {n_subjects} · "
          f"방문 {n_visits} · 시드 {seed}")
    print(f"[synthetic] microbiome {n_msp}  0비율 {np.mean(MIC == 0):.3f}")
    print(f"[synthetic] metabolome {n_metab}  결측 {np.mean(np.isnan(MET)):.4f}  "
          f"(블록 결측 피험자: {', '.join(sorted(failed))})")
    print(f"[synthetic] proteome   {n_prot}  결측 {np.mean(np.isnan(PRO)):.4f}  "
          f"(LOD={lod} 아래 좌측중도절단)")
    print(f"[synthetic] enterotype {md['enteroType'].value_counts().to_dict()}")
    print(f"[synthetic] 층위 신호 세기 effect_scale={effect_scale}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--subjects", type=int, default=69)
    ap.add_argument("--visits", type=int, default=4)
    ap.add_argument("--effect-scale", type=float, default=1.2,
                    help="층위 신호의 세기. 1.0 이 기본, 낮추면 구조가 사라집니다")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.out, n_subjects=a.subjects, n_visits=a.visits,
         effect_scale=a.effect_scale, seed=a.seed)
