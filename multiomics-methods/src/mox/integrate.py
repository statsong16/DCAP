"""
integrate.py — 여러 블록을 하나의 표본 표현으로 합칩니다.

문제: 블록마다 feature 수가 다릅니다. 전처리한 행렬을 그대로 이어붙이면
feature 가 많은 블록이 거리를 독점합니다. 마이크로바이옴 1350개와 대사체
413개를 이어붙이면 사실상 마이크로바이옴만 보는 군집이 됩니다.

여기서 쓰는 방법: **블록별로 같은 수의 주성분을 뽑아 이어붙입니다.**
블록마다 같은 지분(k차원)을 갖게 되고, 전처리·PCA 가 전부 fit/transform 이라
fold 안에서만 학습시키기 쉽습니다.

더 정교한 대안(MOFA, SNF, DIABLO)이 있지만 여기 넣지 않았습니다.
표본이 수백 개인 자료에서 그 복잡도를 얹을 근거가 먼저 있어야 하고,
블록별 PCA 는 그 근거를 재는 기준선 역할을 합니다. 기준선을 이기지 못하면
복잡한 방법을 쓸 이유가 없습니다.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from mox.preprocess import DEFAULT_KIND, BlockPrep


class MultiOmicsEmbedder:
    """
    블록별 전처리 -> 블록별 PCA -> 이어붙이기.

    파라미터
      n_components  블록당 주성분 수 (블록마다 같은 지분)
      kind_map      블록 이름 -> preprocess.KINDS
      prep_kwargs   BlockPrep 에 넘길 추가 인자 (na_strategy, lod 등)
    """

    def __init__(self, n_components: int = 10, seed: int = 0,
                 kind_map: dict | None = None, prep_kwargs: dict | None = None):
        self.n_components = n_components
        self.seed = seed
        self.kind_map = {**DEFAULT_KIND, **(kind_map or {})}
        self.prep_kwargs = prep_kwargs or {}

    def fit(self, blocks: dict, idx: np.ndarray | None = None) -> "MultiOmicsEmbedder":
        """idx 는 train 표본의 인덱스입니다. None 이면 전체 (기술 통계용)."""
        self.block_names_ = list(blocks)
        self.preps_, self.pcas_, self.var_explained_ = {}, {}, {}
        for name in self.block_names_:
            X = self._subset(blocks[name], idx)
            if name not in self.kind_map:
                raise ValueError(f"'{name}' 의 kind 를 모릅니다. kind_map 으로 지정하세요")
            prep = BlockPrep(self.kind_map[name], **self.prep_kwargs).fit(X)
            Z = prep.transform(X)
            k = min(self.n_components, Z.shape[0] - 1, Z.shape[1])
            if k < 1:
                raise ValueError(f"{name}: 주성분을 뽑기에 표본이 부족합니다")
            pca = PCA(n_components=k, random_state=self.seed).fit(Z)
            self.preps_[name], self.pcas_[name] = prep, pca
            self.var_explained_[name] = float(pca.explained_variance_ratio_.sum())
        return self

    def transform(self, blocks: dict, idx: np.ndarray | None = None) -> np.ndarray:
        per = self.transform_per_block(blocks, idx)
        return np.hstack([per[name] for name in self.block_names_])

    def transform_per_block(self, blocks: dict,
                            idx: np.ndarray | None = None) -> dict:
        """블록 하나만 쓰는 베이스라인용. 결합이 실제로 이득인지 재려면 필요합니다."""
        self._check_fitted()
        return {name: self.pcas_[name].transform(
            self.preps_[name].transform(self._subset(blocks[name], idx)))
            for name in self.block_names_}

    def fit_transform(self, blocks: dict, idx: np.ndarray | None = None) -> np.ndarray:
        return self.fit(blocks, idx).transform(blocks, idx)

    def summary(self) -> str:
        self._check_fitted()
        lines = [f"결합 임베딩 {self.n_dim_} 차원 (블록당 {self.n_components}개)"]
        for name in self.block_names_:
            lines.append(f"  {name:12s} feature {self.preps_[name].n_features_out:5d} -> "
                         f"PC {self.pcas_[name].n_components_:2d}  "
                         f"설명 분산 {100 * self.var_explained_[name]:5.1f}%")
        return "\n".join(lines)

    @property
    def n_dim_(self) -> int:
        self._check_fitted()
        return int(sum(p.n_components_ for p in self.pcas_.values()))

    def _check_fitted(self):
        if not hasattr(self, "preps_"):
            raise RuntimeError("fit 을 먼저 호출하세요")

    @staticmethod
    def _subset(df, idx):
        X = df.to_numpy(dtype=float) if hasattr(df, "to_numpy") else np.asarray(df, float)
        return X if idx is None else X[idx]
