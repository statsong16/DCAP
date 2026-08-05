"""결측 진단이 세 가지 기전을 실제로 구분하는지."""

import numpy as np
import pandas as pd

from mox.qc import (censoring_report, flag_incomplete, report,
                    sample_missingness, subject_missingness)


def make_md(n_subjects=6, n_visits=4):
    rows = [{"sample.ID": f"S{i}_v{v}", "metadata.ID": f"S{i}"}
            for i in range(n_subjects) for v in range(n_visits)]
    return pd.DataFrame(rows)


def test_censoring_report_detects_left_censoring():
    """낮은 값일수록 자주 결측 -> 검출한계로 판정되어야 합니다."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(200, 40)) + np.linspace(-3, 3, 40)
    X[X < -1.5] = np.nan                       # LOD 아래를 잘라냄
    rep = censoring_report(X)
    assert rep["likely_left_censored"]
    assert rep["corr_na_vs_mean"] < -0.15


def test_censoring_report_does_not_flag_random_missing():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, size=(200, 40)) + np.linspace(-3, 3, 40)
    X[rng.random(X.shape) < 0.05] = np.nan     # 값과 무관한 결측
    rep = censoring_report(X)
    assert not rep["likely_left_censored"]


def test_censoring_report_handles_no_missing():
    X = np.random.default_rng(2).normal(size=(30, 5))
    rep = censoring_report(X)
    assert rep["na_rate_overall"] == 0.0
    assert not rep["likely_left_censored"]


def test_subject_missingness_finds_concentrated_failure():
    """
    표본 단위로 보면 흩어져 보이지만 피험자 단위로는 한 명에게 몰린 결측.
    실제 자료에서 관찰한 모양입니다.
    """
    md = make_md()
    X = np.zeros((len(md), 50))
    bad = md["metadata.ID"] == "S3"
    X[bad.to_numpy(), :35] = np.nan            # S3 의 모든 방문에서 70% 결측
    blocks = {"metabolome": pd.DataFrame(X)}

    subj = subject_missingness(blocks, md).set_index("metadata.ID")
    assert subj.loc["S3", "metabolome"] == 0.70
    assert (subj.drop(index="S3")["metabolome"] == 0).all()


def test_flag_incomplete_catches_all_visits_of_failed_subject():
    md = make_md()
    X = np.zeros((len(md), 50))
    bad = (md["metadata.ID"] == "S3").to_numpy()
    X[bad, :35] = np.nan
    blocks = {"metabolome": pd.DataFrame(X)}

    flag = flag_incomplete(blocks, md, max_missing=0.20)
    assert flag.sum() == 4                     # 4회 방문 전부
    assert (flag == bad).all()


def test_flag_incomplete_threshold_is_respected():
    md = make_md()
    X = np.zeros((len(md), 100))
    X[:, :10] = np.nan                         # 모든 표본이 10% 결측
    blocks = {"metabolome": pd.DataFrame(X)}
    assert flag_incomplete(blocks, md, max_missing=0.20).sum() == 0
    assert flag_incomplete(blocks, md, max_missing=0.05).sum() == len(md)


def test_sample_missingness_shape_and_columns():
    md = make_md()
    blocks = {"a": pd.DataFrame(np.zeros((len(md), 5))),
              "b": pd.DataFrame(np.zeros((len(md), 5)))}
    s = sample_missingness(blocks, md)
    assert list(s.columns) == ["sample.ID", "metadata.ID", "a", "b"]
    assert len(s) == len(md)


def test_report_runs_and_mentions_blocks():
    md = make_md()
    X = np.random.default_rng(3).normal(size=(len(md), 20))
    X[0, :5] = np.nan
    blocks = {"proteome": pd.DataFrame(X)}
    text = report(blocks, md)
    assert "proteome" in text and "결측" in text
