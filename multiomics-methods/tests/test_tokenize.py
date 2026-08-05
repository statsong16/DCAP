"""토큰화: CLR 성질, 순위 인코딩, 구간 경계의 train-only 학습, 계통 마스크."""

import numpy as np
import pandas as pd
import pytest

from mox.tokenize import (N_SPECIAL, PAD_ID, ValueEncoder, attention_mask, clr,
                          lineage_adjacency, rank_tokenize, sparsity,
                          to_relative)


def test_to_relative_rows_sum_to_one():
    X = np.random.default_rng(0).random((10, 7))
    assert np.allclose(to_relative(X).sum(axis=1), 1.0)


def test_to_relative_handles_all_zero_row():
    X = np.zeros((3, 4))
    assert np.isfinite(to_relative(X)).all()


def test_clr_rows_sum_to_zero():
    X = np.random.default_rng(1).random((8, 15))
    assert np.allclose(clr(X).sum(axis=1), 0.0, atol=1e-9)


def test_clr_is_scale_invariant():
    base = np.random.default_rng(2).random(20) + 0.01
    X = np.vstack([base, base * 5.0])
    Z = clr(X)
    assert np.allclose(Z[0], Z[1], atol=1e-9)


def test_rank_tokenize_orders_by_abundance():
    X = np.array([[0.1, 0.5, 0.0, 0.4]])
    ids, ranks = rank_tokenize(X, max_len=4)
    # 풍부도 순: feature 1 (0.5) > 3 (0.4) > 0 (0.1), feature 2 는 0 이라 빠짐
    assert ids[0, 0] == 1 + N_SPECIAL
    assert ids[0, 1] == 3 + N_SPECIAL
    assert ids[0, 2] == 0 + N_SPECIAL
    assert ids[0, 3] == PAD_ID
    assert list(ranks[0, :3]) == [0, 1, 2]


def test_rank_tokenize_truncates_to_max_len():
    X = np.random.default_rng(3).random((5, 50)) + 0.01
    ids, _ = rank_tokenize(X, max_len=10)
    assert ids.shape == (5, 10)
    assert (ids >= N_SPECIAL).all()


def test_value_encoder_bins_are_learned_on_train_only():
    """
    구간 경계를 전체 자료에서 정하면 held-out 분포가 경계를 통해 새어 들어옵니다.
    train 만으로 fit 한 경계와 전체로 fit 한 경계는 달라야 합니다.
    """
    rng = np.random.default_rng(4)
    train = rng.normal(0, 1, size=(30, 10))
    test = rng.normal(8, 1, size=(30, 10))
    e_train = ValueEncoder("bin", n_bins=8).fit(train)
    e_all = ValueEncoder("bin", n_bins=8).fit(np.vstack([train, test]))
    assert not np.allclose(e_train.edges_, e_all.edges_)


def test_value_encoder_rank_mode_ignores_values():
    X = np.random.default_rng(5).random((4, 6)) + 0.01
    ids, _ = rank_tokenize(to_relative(X), max_len=6)
    out = ValueEncoder("rank").fit_transform(clr(X), ids)
    assert (out == 0).all()


def test_value_encoder_pads_are_zero():
    X = np.array([[0.5, 0.5, 0.0, 0.0]])
    ids, _ = rank_tokenize(X, max_len=4)
    out = ValueEncoder("cont").fit_transform(clr(X), ids)
    assert out[0, 2] == 0 and out[0, 3] == 0


def test_value_encoder_requires_fit():
    with pytest.raises(RuntimeError):
        ValueEncoder("bin").transform(np.zeros((2, 3)), np.zeros((2, 3), dtype=int))


def test_bad_mode_raises():
    with pytest.raises(ValueError):
        ValueEncoder("nonsense")


def test_lineage_adjacency_groups_same_family():
    lin = pd.DataFrame({"taxon": ["a", "b", "c"],
                        "family": ["F1", "F1", "F2"]})
    A = lineage_adjacency(lin, ["a", "b", "c"], level="family")
    assert A[0, 1] == 1 and A[1, 0] == 1
    assert A[0, 2] == 0
    assert A[2, 2] == 1


def test_lineage_adjacency_unknown_features_are_isolated():
    lin = pd.DataFrame({"taxon": ["a"], "family": ["F1"]})
    A = lineage_adjacency(lin, ["a", "zzz", "yyy"], level="family")
    assert A[1, 2] == 0, "정보 없는 feature 끼리 묶이면 안 됩니다"
    assert A[1, 1] == 1


def test_lineage_adjacency_missing_level_raises():
    lin = pd.DataFrame({"taxon": ["a"], "family": ["F1"]})
    with pytest.raises(ValueError):
        lineage_adjacency(lin, ["a"], level="genus")


def test_attention_mask_never_fully_blocks_a_row():
    """완전히 막힌 행이 생기면 softmax 가 NaN 이 됩니다."""
    rng = np.random.default_rng(6)
    X = rng.random((4, 10)) + 0.01
    ids, _ = rank_tokenize(to_relative(X), max_len=6)
    A = np.eye(10, dtype=np.float32)                 # 자기 자신만 허용 (가장 촘촘)
    mask = attention_mask(ids, A)
    assert mask.any(axis=2).all(), "전부 막힌 행이 있습니다"


def test_attention_mask_shape():
    X = np.random.default_rng(7).random((3, 8)) + 0.01
    ids, _ = rank_tokenize(to_relative(X), max_len=5)
    mask = attention_mask(ids, np.eye(8, dtype=np.float32))
    assert mask.shape == (3, 5, 5)


def test_sparsity_bounds():
    assert sparsity(np.ones((4, 4))) == 1.0
    assert sparsity(np.eye(4)) == pytest.approx(0.25)
