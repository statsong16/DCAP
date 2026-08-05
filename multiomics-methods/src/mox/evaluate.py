"""
evaluate.py — 판별력과 **보정**을 함께 잽니다.

AUC 만 보고하는 것은 결과가 아닙니다. AUC 는 순위만 보므로, 예측확률이
전부 0.9 쪽으로 쏠려 있어도 순서만 맞으면 높게 나옵니다. 그 확률을 위험도로
쓰는 순간 틀립니다.

  Brier            확률 예측의 평균제곱오차. 낮을수록 좋음.
  보정기울기       logit(p) 에 대한 y 의 로지스틱 회귀 기울기.
                   1 = 잘 보정됨. 1보다 작으면 과신(예측이 너무 극단적).
                   0에 가까우면 예측확률에 정보가 거의 없다는 뜻.

pooled OOF 규약: fold 마다 예측을 모아 마지막에 한 번 채점합니다.
fold 별 지표의 평균이 아니라서, 소수 클래스가 fold 에 몇 개 없을 때
지표가 정의되지 않는 문제를 피할 수 있습니다.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, mean_absolute_error, r2_score,
                             roc_auc_score)


def calibration_slope(y: np.ndarray, p: np.ndarray) -> float:
    """logit(p) 에 대한 y 의 로지스틱 회귀 기울기. 1 이면 잘 보정된 것."""
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p)).reshape(-1, 1)
    if np.std(x) < 1e-9 or len(np.unique(y)) < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = LogisticRegression(penalty=None, max_iter=1000).fit(x, y)
    return float(lr.coef_[0][0])


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    return {"AUC": float(roc_auc_score(y, p)),
            "Brier": float(brier_score_loss(y, p)),
            "calib_slope": calibration_slope(y, p)}


def regression_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return {"R2": float(r2_score(y, p)), "MAE": float(mean_absolute_error(y, p))}


class OutOfFoldCollector:
    """
    fold 별 예측을 모아 마지막에 한 번 채점합니다.

    모든 표본이 정확히 한 번씩 held-out 이었는지 확인하는 책임도 여기 있습니다.
    이 확인이 없으면 일부 표본이 빠지거나 두 번 들어가도 조용히 지나갑니다.
    """

    def __init__(self, n: int):
        self.n = int(n)
        self._pred = {}
        self._filled = np.zeros(self.n, dtype=int)

    def add(self, key, test_idx: np.ndarray, values):
        if key not in self._pred:
            self._pred[key] = np.full(self.n, np.nan)
        self._pred[key][test_idx] = values

    def mark(self, test_idx: np.ndarray):
        """이번 fold 에서 held-out 이었던 표본을 기록합니다."""
        self._filled[test_idx] += 1

    def check(self):
        if not np.all(self._filled == 1):
            bad = np.where(self._filled != 1)[0]
            raise AssertionError(
                f"표본 {len(bad)}개가 정확히 한 번 held-out 이 아닙니다 "
                f"(예: 인덱스 {bad[:5].tolist()}, 횟수 {self._filled[bad[:5]].tolist()})")

    def predictions(self) -> dict:
        self.check()
        return dict(self._pred)
