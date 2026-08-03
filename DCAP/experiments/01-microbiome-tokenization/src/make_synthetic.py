"""
make_synthetic.py — cMD 없이도 파이프라인이 도는지 확인하는 용도.

R 스크립트가 만드는 것과 같은 형식의 CSV 4개를 생성합니다.
연구(study) 간 배치효과와 질환 신호를 일부러 넣어 두었으므로,
downstream 에서 study 단위 교차검증이 왜 필요한지도 그대로 재현됩니다.

실제 분석에서는 절대 쓰지 마세요. 파이프라인 점검 전용입니다.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def main(out_dir: str, n_per_study: int = 120, n_taxa: int = 180,
         n_path: int = 90, n_studies: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    families = [f"Family_{i:02d}" for i in range(18)]
    fam_of = rng.integers(0, len(families), size=n_taxa)
    taxa_names = [f"s__Species_{i:03d}" for i in range(n_taxa)]
    path_names = [f"PWY-{i:04d}" for i in range(n_path)]

    # 과(family) 단위 기본 로그풍부도 -> 계통 구조가 실제로 존재하게 만듦
    fam_base = rng.normal(0, 1.0, size=len(families))
    base = fam_base[fam_of] + rng.normal(0, 0.4, size=n_taxa)

    # 질환 신호: 특정 과에 속한 분류군만 이동
    sick_fams = rng.choice(len(families), size=4, replace=False)
    effect = np.where(np.isin(fam_of, sick_fams), 0.9, 0.0)

    rows_t, rows_p, meta = [], [], []
    for s in range(n_studies):
        study = f"Study_{chr(65 + s)}"
        batch = rng.normal(0, 0.6, size=n_taxa)         # 연구별 배치효과
        batch_p = rng.normal(0, 0.6, size=n_path)
        for j in range(n_per_study):
            sid = f"{study}_S{j:04d}"
            y = int(rng.random() < 0.45)
            log_ab = base + batch + effect * y + rng.normal(0, 0.7, size=n_taxa)
            ab = np.exp(log_ab)
            ab *= (rng.random(n_taxa) > 0.35)            # 희소성 (0이 다수)
            if ab.sum() == 0:
                ab[rng.integers(0, n_taxa)] = 1.0
            ab = ab / ab.sum() * 100.0

            # pathway 는 taxa 에서 파생시켜 두 모달리티가 실제로 상관되게 함
            mix = rng.normal(0, 1, size=(n_taxa, n_path)) * (rng.random((n_taxa, n_path)) < 0.05)
            pw = np.exp(np.log1p(ab) @ mix / np.sqrt(n_taxa) + batch_p
                        + rng.normal(0, 0.5, size=n_path))
            pw *= (rng.random(n_path) > 0.2)
            if pw.sum() == 0:
                pw[rng.integers(0, n_path)] = 1.0
            pw = pw / pw.sum() * 100.0

            rows_t.append([sid] + ab.tolist())
            rows_p.append([sid] + pw.tolist())
            meta.append({
                "sample_id": sid, "study_name": study,
                "study_condition": "case" if y else "control",
                "disease": "target" if y else "healthy",
                "age": int(rng.normal(66, 8)),
                "gender": rng.choice(["male", "female"]),
            })

    pd.DataFrame(rows_t, columns=["sample_id"] + taxa_names).to_csv(
        os.path.join(out_dir, "taxa_abundance.csv"), index=False)
    pd.DataFrame(rows_p, columns=["sample_id"] + path_names).to_csv(
        os.path.join(out_dir, "pathway_abundance.csv"), index=False)
    pd.DataFrame(meta).to_csv(os.path.join(out_dir, "metadata.csv"), index=False)
    pd.DataFrame({
        "kingdom": "Bacteria",
        "phylum": [f"Phylum_{f // 4}" for f in fam_of],
        "class": [f"Class_{f // 3}" for f in fam_of],
        "order": [f"Order_{f // 2}" for f in fam_of],
        "family": [families[f] for f in fam_of],
        "genus": [f"Genus_{i // 3:02d}" for i in range(n_taxa)],
        "species": taxa_names,
        "taxon": taxa_names,
    }).to_csv(os.path.join(out_dir, "taxa_lineage.csv"), index=False)

    print(f"합성 데이터 생성 완료 -> {out_dir}")
    print(f"  샘플 {n_per_study * n_studies}개 / taxa {n_taxa} / pathway {n_path} / study {n_studies}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-per-study", type=int, default=120)
    ap.add_argument("--n-studies", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(out_dir=a.out, n_per_study=a.n_per_study, n_studies=a.n_studies, seed=a.seed)
