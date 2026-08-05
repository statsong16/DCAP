"""전처리가 train 에서만 학습되는지, 결측 대치가 기전에 맞게 동작하는지."""

import numpy as np
import pytest

from mox.preprocess import BlockPrep, make_preps


def test_fit_does_not_see_test_rows():
    """
    누수 테스트: train 만 보고 fit 한 결과가, test 를 섞어 fit 한 결과와 달라야
    합니다. 같다면 transform 이 어딘가에서 전체 자료를 다시 보고 있다는 뜻입니다.
    """
    rng = np.random.default_rng(0)
    Xtr = rng.normal(0, 1, size=(40, 6))
    Xte = rng.normal(10, 1, size=(40, 6))            # 분포가 확 다른 test

    p_train = BlockPrep("log_abundance").fit(Xtr)
    p_all = BlockPrep("log_abundance").fit(np.vstack([Xtr, Xte]))
    assert not np.allclose(p_train.mean_, p_all.mean_)

    # train 만으로 fit 했으면 test 의 평균은 0 근처가 아니어야 정상입니다.
    Zte = p_train.transform(Xte)
    assert Zte.mean() > 3, "test 가 자기 자신의 통계로 표준화된 것 같습니다"


def test_transform_is_deterministic_given_fit():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 5))
    p = BlockPrep("log_abundance").fit(X)
    assert np.allclose(p.transform(X), p.transform(X))


def test_left_censored_fills_lower_than_median():
    """검출한계 대치는 중앙값 대치보다 반드시 낮은 값을 채워야 합니다."""
    rng = np.random.default_rng(2)
    X = rng.normal(5.0, 1.0, size=(60, 4))
    X[:10, 0] = np.nan                                # 한 feature 에 결측

    med = BlockPrep("log_abundance", na_strategy="median").fit(X)
    cen = BlockPrep("log_abundance", na_strategy="left_censored").fit(X)
    assert cen.fill_[0] < med.fill_[0]


def test_explicit_lod_takes_precedence():
    rng = np.random.default_rng(3)
    X = rng.normal(5.0, 1.0, size=(50, 3))
    X[:8, 1] = np.nan
    lod = np.array([-9.0, -9.0, -9.0])
    p = BlockPrep("log_abundance", na_strategy="left_censored", lod=lod).fit(X)
    assert p.fill_[1] == pytest.approx(-9.0)


def test_compositional_clr_rows_sum_to_zero():
    """CLR 은 정의상 행 합이 0 입니다. 아니면 폐쇄성을 풀지 못한 것입니다."""
    rng = np.random.default_rng(4)
    X = rng.random((25, 12)) * (rng.random((25, 12)) > 0.3)
    X[X.sum(axis=1) == 0, 0] = 1.0
    p = BlockPrep("compositional", min_prevalence=0.0).fit(X)
    Z = p._to_value(X)
    assert np.allclose(Z.sum(axis=1), 0.0, atol=1e-8)


def test_compositional_is_invariant_to_sequencing_depth():
    """
    같은 조성인데 총량만 다른 두 표본은 같은 CLR 값을 가져야 합니다.
    TSS 를 빠뜨리면 여기서 깨집니다.
    """
    rng = np.random.default_rng(5)
    base = rng.random(20) + 0.01
    X = np.vstack([base, base * 7.0, base * 0.3])
    p = BlockPrep("compositional", min_prevalence=0.0).fit(X)
    Z = p._to_value(X)
    assert np.allclose(Z[0], Z[1], atol=1e-8)
    assert np.allclose(Z[0], Z[2], atol=1e-8)


def test_zero_variance_features_are_dropped():
    X = np.random.default_rng(6).normal(size=(20, 4))
    X[:, 2] = 3.0                                     # 상수 feature
    p = BlockPrep("log_abundance").fit(X)
    assert p.n_features_out == 3


def test_high_missing_features_are_dropped():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(50, 5))
    X[:40, 3] = np.nan                                # 80% 결측
    p = BlockPrep("log_abundance", max_na_frac=0.2).fit(X)
    assert p.keep_[3] == np.False_


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        BlockPrep("log_abundance").transform(np.zeros((3, 3)))


def test_bad_kind_raises():
    with pytest.raises(ValueError):
        BlockPrep("nonsense")


def test_make_preps_uses_defaults():
    preps = make_preps(["microbiome", "metabolome", "proteome"])
    assert preps["microbiome"].kind == "compositional"
    assert preps["metabolome"].kind == "intensity"
    # NPX 는 검출한계 결측이 기본이어야 합니다
    assert preps["proteome"].na_strategy == "left_censored"


def test_make_preps_unknown_block_raises():
    with pytest.raises(ValueError):
        make_preps(["transcriptome"])
