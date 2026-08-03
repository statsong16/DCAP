"""
data.py — 3개 오믹스 블록의 전처리와 결합 임베딩.

분할 전략 (CLAUDE.md §6):
  이 자료는 피험자 69명의 4회 반복측정입니다. 표본 단위로 나누면 같은 사람이
  train 과 test 양쪽에 들어가고, 모델은 "이 사람"을 외우면 됩니다.
  Age·Gender·BMI 는 사실상 피험자 상수이므로(피험자 내 SD: Age 0.34년, BMI 0.30)
  표본 단위 분할에서는 성능이 거의 자동으로 부풀려집니다.
  따라서 모든 분할은 metadata.ID 기준 GroupKFold 입니다.

파라미터를 학습하는 전처리 — prevalence 필터, pseudocount, 결측 대치값,
스케일러, PCA — 는 전부 train fold 에서만 fit 하고 held-out 에 적용합니다.
그래서 전처리가 함수가 아니라 fit/transform 객체로 되어 있습니다.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

BLOCKS = ("microbiome", "metabolome", "proteome")

# 블록별 처리 방식. 값의 성격이 달라서 같은 변환을 쓸 수 없습니다.
BLOCK_KIND = {
    "microbiome": "compositional",   # 희소 상대풍부도 -> TSS -> CLR
    "metabolome": "intensity",       # 원시 강도(3e3~1e11) -> log10
    "proteome":   "npx",             # Olink NPX. 이미 log2 이므로 변환 안 함
}


def load_blocks(data_dir: str):
    """
    CSV 4개를 읽어 (blocks, metadata) 를 돌려줍니다.

    CSV 는 실제 RData 와 같은 방향(행=feature, 열=sample)으로 저장되어 있으므로
    여기서 전치해 samples x features 로 맞춥니다.
    """
    md = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    blocks = {}
    for b in BLOCKS:
        df = pd.read_csv(os.path.join(data_dir, f"{b}.csv"), index_col=0).T
        # 표본 순서를 metadata 에 맞춥니다. RData 는 이미 정렬돼 있지만 믿지 않습니다.
        missing = set(md["sample.ID"]) - set(df.index)
        if missing:
            raise ValueError(f"{b}: metadata 에 있는 표본 {len(missing)}개가 없습니다")
        blocks[b] = df.loc[md["sample.ID"]].astype(float)
    return blocks, md


class BlockPrep:
    """
    한 블록의 전처리. fit 은 train fold 에서만 호출합니다.

    남기는 feature 집합, pseudocount, 대치용 중앙값, 평균/표준편차가 전부
    fit 시점에 고정되고 transform 은 그것을 그대로 적용하기만 합니다.
    """

    def __init__(self, kind: str, min_prevalence: float = 0.10,
                 max_na_frac: float = 0.20):
        self.kind = kind
        self.min_prevalence = min_prevalence
        self.max_na_frac = max_na_frac

    def fit(self, X: np.ndarray) -> "BlockPrep":
        X = np.asarray(X, dtype=np.float64)
        if self.kind == "compositional":
            # 존재율이 낮은 taxa 는 0/1 잡음에 가깝습니다. train 에서만 판단합니다.
            self.keep_ = (np.nan_to_num(X) > 0).mean(axis=0) >= self.min_prevalence
            nz = X[:, self.keep_]
            nz = nz[nz > 0]
            self.pseudo_ = (nz.min() / 2.0) if nz.size else 1e-9
        else:
            self.keep_ = np.isnan(X).mean(axis=0) <= self.max_na_frac
        if self.keep_.sum() == 0:
            raise ValueError(f"{self.kind}: 남는 feature 가 없습니다")

        Z = self._raw_to_value(X)
        self.median_ = np.nanmedian(Z, axis=0)
        self.median_ = np.where(np.isnan(self.median_), 0.0, self.median_)
        Z = np.where(np.isnan(Z), self.median_, Z)

        self.mean_ = Z.mean(axis=0)
        self.std_ = Z.std(axis=0)
        # 분산이 0인 feature 는 train 에서 정보가 없으므로 뺍니다.
        self.var_ok_ = self.std_ > 1e-12
        self.std_ = np.where(self.var_ok_, self.std_, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = self._raw_to_value(np.asarray(X, dtype=np.float64))
        Z = np.where(np.isnan(Z), self.median_, Z)
        Z = (Z - self.mean_) / self.std_
        return Z[:, self.var_ok_]

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    # -- 블록 종류별 값 변환 (feature 선택 이후, 표준화 이전) ----------------
    def _raw_to_value(self, X: np.ndarray) -> np.ndarray:
        Xk = X[:, self.keep_]
        if self.kind == "compositional":
            # 열 합이 표본마다 다르므로(실제 자료 CV 0.12) 먼저 TSS 로 맞춥니다.
            s = np.nansum(Xk, axis=1, keepdims=True)
            s[s <= 0] = 1.0
            R = np.nan_to_num(Xk) / s
            L = np.log(R + self.pseudo_)
            # CLR: 조성의 폐쇄성(합=1)을 풀지 않으면 가짜 음의 상관이 그대로 들어옵니다.
            return L - L.mean(axis=1, keepdims=True)
        if self.kind == "intensity":
            # 6자리 이상 퍼진 강도. 0은 검출 한계 아래이므로 +1 후 로그.
            return np.log10(np.where(Xk < 0, 0.0, Xk) + 1.0)
        if self.kind == "npx":
            return Xk        # NPX 는 이미 log2 스케일
        raise ValueError(f"알 수 없는 블록 종류: {self.kind}")


class MultiOmicsEmbedder:
    """
    블록별 전처리 -> 블록별 PCA -> 이어붙이기.

    블록마다 feature 수가 1350/413/770 으로 달라서 그대로 이어붙이면
    microbiome 이 거리를 독점합니다. 블록당 같은 수의 주성분으로 맞춰
    세 블록이 같은 지분을 갖게 합니다. (SNF/MOFA 같은 방법도 있지만
    표본 276개에 그 복잡도를 얹을 근거가 아직 없습니다 — CLAUDE.md §2)
    """

    def __init__(self, n_components: int = 10, seed: int = 0):
        self.n_components = n_components
        self.seed = seed

    def fit(self, blocks: dict, idx: np.ndarray) -> "MultiOmicsEmbedder":
        self.preps_, self.pcas_, self.var_explained_ = {}, {}, {}
        for b, df in blocks.items():
            Xtr = df.to_numpy()[idx]
            prep = BlockPrep(BLOCK_KIND[b]).fit(Xtr)
            Ztr = prep.transform(Xtr)
            k = min(self.n_components, Ztr.shape[0] - 1, Ztr.shape[1])
            pca = PCA(n_components=k, random_state=self.seed).fit(Ztr)
            self.preps_[b], self.pcas_[b] = prep, pca
            self.var_explained_[b] = float(pca.explained_variance_ratio_.sum())
        return self

    def transform(self, blocks: dict, idx: np.ndarray | None = None) -> np.ndarray:
        out = []
        for b, df in blocks.items():
            X = df.to_numpy() if idx is None else df.to_numpy()[idx]
            out.append(self.pcas_[b].transform(self.preps_[b].transform(X)))
        return np.hstack(out)

    def transform_per_block(self, blocks: dict, idx: np.ndarray | None = None) -> dict:
        """블록 하나만 쓰는 베이스라인용."""
        return {b: self.pcas_[b].transform(
            self.preps_[b].transform(df.to_numpy() if idx is None else df.to_numpy()[idx]))
            for b, df in blocks.items()}


def subject_folds(md: pd.DataFrame, n_splits: int = 5, group_col: str = "metadata.ID"):
    """
    피험자 단위 GroupKFold. (train_idx, test_idx) 를 순서대로 내놓습니다.
    한 피험자의 4회 방문은 항상 같은 쪽에 있습니다.
    """
    groups = md[group_col].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(np.zeros(len(md)), groups=groups):
        assert not (set(groups[tr]) & set(groups[te])), "피험자가 양쪽 fold 에 있습니다"
        yield tr, te
