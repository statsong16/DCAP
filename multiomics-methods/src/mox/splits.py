"""
splits.py — 교차검증 분할. 이 파일이 이 저장소에서 제일 중요합니다.

코호트 자료는 한 사람에게서 여러 번, 한 연구에서 여러 명이 나옵니다.
표본을 무작위로 나누면 같은 사람(또는 같은 배치)이 train 과 test 양쪽에
들어가고, 모델은 표현형 대신 **그 사람**을 외워서 맞힙니다. 성능은 올라가고
결론은 틀립니다.

특히 표적이 피험자 상수(성별, 진단, 기저 특성)에 가까울수록 부풀림이 큽니다.
그때 표본 단위 분할의 성능은 사실상 "재식별 정확도"입니다.

  올바름:  GroupKFold(groups=피험자)      -> subject_folds
  대조용:  KFold(표본)                    -> leaky_folds  (보고용 아님)

leaky_folds 를 굳이 넣어 둔 이유는, 누수의 크기를 주장하지 말고 재보라는
것입니다. 같은 자료 같은 모델에서 분할만 바꿔 두 수치를 나란히 놓으면
누구에게든 설명이 끝납니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from mox.io import SUBJECT_COL


def subject_folds(md: pd.DataFrame, n_splits: int = 5,
                  group_col: str = SUBJECT_COL):
    """
    피험자 단위 GroupKFold. (train_idx, test_idx) 를 차례로 내놓습니다.
    한 피험자의 모든 표본은 항상 같은 쪽에 있습니다.
    """
    groups = md[group_col].to_numpy()
    n_groups = len(np.unique(groups))
    if n_groups < n_splits:
        raise ValueError(f"그룹이 {n_groups}개뿐이라 {n_splits}-fold 를 만들 수 없습니다")
    for tr, te in GroupKFold(n_splits=n_splits).split(np.zeros(len(md)),
                                                      groups=groups):
        assert_no_group_overlap(groups, tr, te)
        yield tr, te


def leaky_folds(md: pd.DataFrame, n_splits: int = 5, seed: int = 0):
    """
    표본 단위 KFold. **일부러 틀린 분할**입니다.

    누수의 크기를 보여주는 대조용으로만 쓰고, 보고하는 표에는 넣지 마세요.
    """
    for tr, te in KFold(n_splits=n_splits, shuffle=True,
                        random_state=seed).split(np.zeros(len(md))):
        yield tr, te


def assert_no_group_overlap(groups: np.ndarray, train_idx: np.ndarray,
                            test_idx: np.ndarray):
    """
    분할이 그룹을 가르지 않았는지 확인합니다.

    새 모델링 스크립트를 쓸 때마다 이걸 부르세요. 누수는 조용히 들어오고
    결과가 좋아 보이기 때문에 사후에 알아채기 어렵습니다.
    """
    shared = set(np.asarray(groups)[train_idx]) & set(np.asarray(groups)[test_idx])
    if shared:
        raise AssertionError(
            f"그룹 {len(shared)}개가 train 과 test 양쪽에 있습니다: "
            f"{sorted(map(str, shared))[:5]}")


def describe_repeated_measures(md: pd.DataFrame, group_col: str = SUBJECT_COL,
                               constant_candidates=()) -> pd.DataFrame:
    """
    분할 전략을 정하기 전에 자료의 반복측정 구조를 봅니다.

    피험자 내 변동이 거의 없는 변수는 사실상 피험자 상수이고, 그런 표적에
    표본 단위 분할을 쓰면 성능은 재식별 정확도가 됩니다.
    """
    rows = [{"metric": "n_samples", "value": float(len(md))},
            {"metric": "n_groups", "value": float(md[group_col].nunique())},
            {"metric": "samples_per_group_median",
             "value": float(md.groupby(group_col).size().median())}]
    for col in constant_candidates:
        if col not in md.columns:
            continue
        s = md.groupby(group_col)[col]
        if pd.api.types.is_numeric_dtype(md[col]):
            within = float(s.std().mean())
            rows.append({"metric": f"{col}__within_group_sd", "value": within})
        else:
            frac = float((s.nunique() == 1).mean())
            rows.append({"metric": f"{col}__constant_within_group_frac",
                         "value": frac})
    return pd.DataFrame(rows)
