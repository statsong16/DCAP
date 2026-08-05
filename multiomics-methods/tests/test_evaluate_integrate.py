"""평가 지표와 결합 임베딩."""

import numpy as np
import pandas as pd
import pytest

from mox.evaluate import (OutOfFoldCollector, binary_metrics, calibration_slope,
                          regression_metrics)
from mox.integrate import MultiOmicsEmbedder


# ---------------------------------------------------------------- evaluate
def test_calibration_slope_is_one_for_well_calibrated():
    """진짜 확률로 생성한 라벨이면 보정기울기가 1 근처여야 합니다."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.98, size=8000)
    y = (rng.random(8000) < p).astype(int)
    assert calibration_slope(y, p) == pytest.approx(1.0, abs=0.15)


def test_calibration_slope_below_one_when_overconfident():
    """예측을 극단으로 밀면(과신) 기울기가 1보다 작아져야 합니다."""
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, size=8000)
    y = (rng.random(8000) < p).astype(int)
    logit = np.log(p / (1 - p))
    over = 1 / (1 + np.exp(-2.5 * logit))          # 더 뾰족하게
    assert calibration_slope(y, over) < 0.85


def test_calibration_slope_nan_for_constant_prediction():
    y = np.array([0, 1, 0, 1])
    assert np.isnan(calibration_slope(y, np.full(4, 0.5)))


def test_binary_metrics_keys_and_ranges():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=200)
    p = rng.random(200)
    m = binary_metrics(y, p)
    assert set(m) == {"AUC", "Brier", "calib_slope"}
    assert 0 <= m["AUC"] <= 1 and 0 <= m["Brier"] <= 1


def test_regression_metrics_perfect_prediction():
    y = np.arange(20, dtype=float)
    m = regression_metrics(y, y)
    assert m["R2"] == pytest.approx(1.0) and m["MAE"] == pytest.approx(0.0)


def test_oof_collector_requires_every_sample_once():
    c = OutOfFoldCollector(6)
    c.mark(np.array([0, 1, 2]))
    with pytest.raises(AssertionError):
        c.check()                                   # 3,4,5 가 아직 안 들어옴
    c.mark(np.array([3, 4, 5]))
    c.check()


def test_oof_collector_catches_double_holdout():
    c = OutOfFoldCollector(4)
    c.mark(np.array([0, 1]))
    c.mark(np.array([1, 2, 3]))                     # 1 번이 두 번
    with pytest.raises(AssertionError):
        c.check()


def test_oof_collector_stores_predictions():
    c = OutOfFoldCollector(4)
    c.mark(np.array([0, 1])); c.add("k", np.array([0, 1]), [0.1, 0.2])
    c.mark(np.array([2, 3])); c.add("k", np.array([2, 3]), [0.3, 0.4])
    assert np.allclose(c.predictions()["k"], [0.1, 0.2, 0.3, 0.4])


# --------------------------------------------------------------- integrate
def make_blocks(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "microbiome": pd.DataFrame(rng.random((n, 30)) * (rng.random((n, 30)) > 0.4)),
        "metabolome": pd.DataFrame(np.exp(rng.normal(10, 1, size=(n, 20)))),
        "proteome": pd.DataFrame(rng.normal(3, 1, size=(n, 25))),
    }


def test_embedder_gives_each_block_equal_share():
    emb = MultiOmicsEmbedder(n_components=4, seed=0).fit(make_blocks())
    Z = emb.transform(make_blocks())
    assert Z.shape[1] == 4 * 3
    for name in emb.block_names_:
        assert emb.pcas_[name].n_components_ == 4


def test_embedder_fit_on_train_only_changes_output():
    blocks = make_blocks(n=60)
    tr = np.arange(30)
    emb_tr = MultiOmicsEmbedder(3, 0).fit(blocks, tr)
    emb_all = MultiOmicsEmbedder(3, 0).fit(blocks)
    a = emb_tr.transform(blocks)
    b = emb_all.transform(blocks)
    assert not np.allclose(a, b), "train 전용 fit 이 전체 fit 과 같은 결과를 냈습니다"


def test_transform_per_block_matches_concatenation():
    blocks = make_blocks()
    emb = MultiOmicsEmbedder(3, 0).fit(blocks)
    per = emb.transform_per_block(blocks)
    joint = emb.transform(blocks)
    assert np.allclose(np.hstack([per[n] for n in emb.block_names_]), joint)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        MultiOmicsEmbedder(3, 0).transform(make_blocks())


def test_unknown_block_name_raises():
    blocks = {"transcriptome": pd.DataFrame(np.random.default_rng(0).random((20, 5)))}
    with pytest.raises(ValueError):
        MultiOmicsEmbedder(2, 0).fit(blocks)


def test_summary_mentions_every_block():
    emb = MultiOmicsEmbedder(3, 0).fit(make_blocks())
    s = emb.summary()
    for name in ("microbiome", "metabolome", "proteome"):
        assert name in s
