"""분할이 그룹을 가르지 않는지. 이 저장소에서 제일 중요한 테스트입니다."""

import numpy as np
import pandas as pd
import pytest

from mox.splits import (assert_no_group_overlap, describe_repeated_measures,
                        leaky_folds, subject_folds)


def make_md(n_subjects=12, n_visits=4):
    rows = []
    for i in range(n_subjects):
        for v in range(n_visits):
            rows.append({"sample.ID": f"S{i}_v{v}", "metadata.ID": f"S{i}",
                         "Age": 50 + i, "Gender": "Female" if i % 2 else "Male"})
    return pd.DataFrame(rows)


def test_subject_folds_never_split_a_subject():
    md = make_md()
    groups = md["metadata.ID"].to_numpy()
    seen_test = np.zeros(len(md), dtype=int)
    for tr, te in subject_folds(md, n_splits=4):
        assert not (set(groups[tr]) & set(groups[te]))
        seen_test[te] += 1
    # 모든 표본이 정확히 한 번씩 held-out
    assert (seen_test == 1).all()


def test_subject_folds_keep_all_visits_together():
    md = make_md()
    for tr, te in subject_folds(md, n_splits=4):
        for subj in md.loc[te, "metadata.ID"].unique():
            rows = md.index[md["metadata.ID"] == subj]
            assert set(rows).issubset(set(te)), "한 피험자의 방문이 쪼개졌습니다"


def test_leaky_folds_really_do_leak():
    """대조군이 실제로 누수를 만드는지 확인합니다. 안 그러면 대조가 안 됩니다."""
    md = make_md()
    groups = md["metadata.ID"].to_numpy()
    overlaps = [bool(set(groups[tr]) & set(groups[te]))
                for tr, te in leaky_folds(md, n_splits=4, seed=0)]
    assert all(overlaps), "표본 단위 분할인데 그룹이 겹치지 않았습니다"


def test_assert_no_group_overlap_raises():
    groups = np.array(["a", "a", "b", "b"])
    with pytest.raises(AssertionError):
        assert_no_group_overlap(groups, np.array([0, 2]), np.array([1, 3]))
    assert_no_group_overlap(groups, np.array([0, 1]), np.array([2, 3]))


def test_too_few_groups_raises():
    md = make_md(n_subjects=3)
    with pytest.raises(ValueError):
        list(subject_folds(md, n_splits=5))


def test_describe_repeated_measures_flags_subject_constants():
    md = make_md()
    d = describe_repeated_measures(md, constant_candidates=("Age", "Gender"))
    got = dict(zip(d["metric"], d["value"]))
    assert got["n_groups"] == 12
    assert got["samples_per_group_median"] == 4
    # Age 는 피험자 내에서 전혀 변하지 않게 만들었으므로 within-group SD 는 0
    assert got["Age__within_group_sd"] == pytest.approx(0.0)
    assert got["Gender__constant_within_group_frac"] == pytest.approx(1.0)
