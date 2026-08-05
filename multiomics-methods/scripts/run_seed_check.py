"""
run_seed_check.py — 연관 검정 결과가 시드 하나의 우연인지 확인합니다.

시드 하나에서 나온 유의성을 그대로 믿으면 안 됩니다. 합성 자료에서는 어떤
변수에 신호를 심었는지 알고 있으므로, 여러 시드에서 돌려 보면 "진짜 신호"와
"이번 판에 우연히 나온 것"이 구분됩니다.

  심어 둔 것: BMI 하나 (make_synthetic.py 에서 층위 -> BMI 로만 연결)
  나머지(Age, Gender, enteroType)에서 유의하게 나오는 것은 전부 거짓양성입니다.

BH 보정을 해도 거짓양성은 남습니다. 보정은 기대 비율을 통제할 뿐이고,
피험자 67명에 검정 5개면 어떤 판에서는 걸립니다. 그래서 시드 하나의 표를
결론으로 쓰지 말라는 것이 이 스크립트의 요지입니다.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PLANTED = "BMI"          # 합성 자료에서 층위와 실제로 연결한 유일한 변수


def one_seed(seed: int, n_boot: int, drop: float, workdir: str) -> pd.DataFrame:
    data = os.path.join(workdir, f"data_{seed}")
    out = os.path.join(workdir, f"run_{seed}")
    env = {**os.environ, "PYTHONPATH": os.path.join(HERE, "..", "src")}
    subprocess.run([sys.executable, os.path.join(HERE, "make_synthetic.py"),
                    "--out", data, "--seed", str(seed)],
                   check=True, capture_output=True, env=env)
    subprocess.run([sys.executable, os.path.join(HERE, "run_stratify.py"),
                    "--data", data, "--out", out, "--seed", str(seed),
                    "--n-boot", str(n_boot), "--drop-incomplete", str(drop)],
                   check=True, capture_output=True, env=env)
    tab = pd.read_csv(os.path.join(out, "association.csv"))
    return tab.assign(seed=seed)


def main(seeds, n_boot: int, drop: float, out_dir: str, alpha: float):
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as workdir:
        allt = pd.concat([one_seed(s, n_boot, drop, workdir) for s in seeds],
                         ignore_index=True)

    wide = allt.pivot_table(index="variable", columns="seed", values="q_BH")
    print(f"BH 보정 q 값 (시드 {list(seeds)})")
    print(wide.to_string(float_format=lambda x: f"{x:.4g}"))

    hit = (wide < alpha)
    rate = hit.mean(axis=1).rename("유의 비율")
    print(f"\nq < {alpha} 인 시드 비율")
    for var, r in rate.items():
        tag = "  <- 심어 둔 신호" if var == PLANTED else (
            "  <- 거짓양성" if r > 0 else "")
        print(f"  {var:12s} {r:.2f}{tag}")

    n_seeds = len(list(seeds))
    fp = hit.drop(index=PLANTED, errors="ignore").to_numpy().sum()
    total = hit.drop(index=PLANTED, errors="ignore").size
    print(f"\n심지 않은 변수에서 나온 유의 결과: {int(fp)} / {total} "
          f"({int(fp) / max(total, 1):.2%})")
    print(f"심어 둔 {PLANTED} 가 유의한 시드: "
          f"{int(hit.loc[PLANTED].sum()) if PLANTED in hit.index else 0} / {n_seeds}")

    wide.to_csv(os.path.join(out_dir, "seed_check_q.csv"))
    allt.to_csv(os.path.join(out_dir, "seed_check_long.csv"), index=False)
    print(f"\n[saved] {out_dir}/seed_check_q.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-boot", type=int, default=5)
    ap.add_argument("--drop-incomplete", type=float, default=0.2)
    ap.add_argument("--out", default="runs/seed_check")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    main(a.seeds, a.n_boot, a.drop_incomplete, a.out, a.alpha)
