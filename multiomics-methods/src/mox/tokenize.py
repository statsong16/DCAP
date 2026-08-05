"""
tokenize.py — 조성 자료를 트랜스포머 토큰 시퀀스로 바꾸는 층.

이 저장소의 다른 모듈이 "표본 x feature 행렬"을 다루는 데 반해, 여기는
표본 하나를 **토큰 시퀀스**로 보는 관점입니다. 단일세포 파운데이션 모델들이
쓰는 세 가지 값 인코딩을 갈아 끼울 수 있게 두었습니다.

  rank   풍부도 순위만 남기고 절대값을 버립니다 (Geneformer 방식).
         배치효과에 강하지만 "얼마나 많은가"를 통째로 잃습니다.
  bin    CLR 값을 분위수 구간으로 이산화합니다 (scGPT 방식).
         구간 경계는 **train 에서만** 정해야 합니다. 그래서 fit/transform 입니다.
  cont   CLR 값을 연속값 그대로 씁니다 (scFoundation 방식).

어느 쪽이든 CLR 을 먼저 겁니다. 조성 자료의 폐쇄성(합=1)을 풀지 않으면
"하나가 늘면 나머지가 준다"는 가짜 음의 상관이 그대로 모델에 들어갑니다.

계통 정보가 있으면 어텐션을 제약할 수 있습니다 (lineage_adjacency).
분류군은 독립 항목이 아니라 계통수 위에 있고, 완전연결 어텐션은 그 구조를
무시한 채 L^2 개의 쌍을 전부 추정합니다. 표본이 적을수록 손해입니다.

torch 를 쓰지 않습니다. 여기서 나온 배열을 그대로 텐서로 감싸면 됩니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PAD_ID = 0
MASK_ID = 1
N_SPECIAL = 2          # 실제 feature 는 2번부터

VALUE_MODES = ("rank", "bin", "cont")


def to_relative(X: np.ndarray) -> np.ndarray:
    """행 합을 1로 맞춥니다."""
    X = np.asarray(X, dtype=np.float64)
    s = X.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return X / s


def clr(X: np.ndarray, pseudo="half_min") -> np.ndarray:
    """
    중심 로그비 변환.

    pseudo: "half_min" 이면 비영 최솟값의 절반, 숫자면 그 값을 씁니다.
    0 을 어떻게 메우느냐가 결과를 좌우하므로 이 선택은 반드시 기록하세요.
    """
    R = to_relative(X)
    if pseudo == "half_min":
        nz = R[R > 0]
        eps = float(nz.min() / 2.0) if nz.size else 1e-9
    else:
        eps = float(pseudo)
    L = np.log(R + eps)
    return L - L.mean(axis=1, keepdims=True)


def rank_tokenize(X_rel: np.ndarray, max_len: int, min_abundance: float = 0.0):
    """
    풍부도 내림차순으로 검출된 feature 를 나열합니다.

    반환
      ids   (N, max_len) feature 인덱스 + N_SPECIAL, 빈 자리는 PAD_ID
      ranks (N, max_len) 0,1,2,... 순위 (위치 인코딩으로 씁니다)
    """
    X_rel = np.asarray(X_rel, dtype=np.float64)
    N, _ = X_rel.shape
    ids = np.full((N, max_len), PAD_ID, dtype=np.int64)
    ranks = np.zeros((N, max_len), dtype=np.int64)
    for i in range(N):
        present = np.where(X_rel[i] > min_abundance)[0]
        present = present[np.argsort(-X_rel[i, present])][:max_len]
        ids[i, :len(present)] = present + N_SPECIAL
        ranks[i, :len(present)] = np.arange(len(present))
    return ids, ranks


class ValueEncoder:
    """
    CLR 값을 모델 입력으로 바꿉니다. 구간 경계를 학습하므로 fit/transform 입니다.

    **fit 은 train fold 에서만.** 전체 자료의 분위수로 구간을 자르면 held-out
    표본의 분포 정보가 경계를 통해 새어 들어옵니다.
    """

    def __init__(self, mode: str = "bin", n_bins: int = 32):
        if mode not in VALUE_MODES:
            raise ValueError(f"mode 는 {VALUE_MODES} 중 하나여야 합니다: {mode!r}")
        self.mode = mode
        self.n_bins = n_bins

    def fit(self, X_clr: np.ndarray) -> "ValueEncoder":
        if self.mode == "bin":
            qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
            self.edges_ = np.quantile(np.asarray(X_clr, float).ravel(), qs)
        else:
            self.edges_ = None
        self._fitted = True
        return self

    def transform(self, X_clr: np.ndarray, ids: np.ndarray) -> np.ndarray:
        """
        ids 순서에 맞춰 값 배열을 만듭니다. PAD 자리는 0 입니다.

        rank 모드는 값을 쓰지 않으므로 0 배열을 돌려줍니다 (순위는 ids 에 있음).
        """
        if not getattr(self, "_fitted", False):
            raise RuntimeError("fit 을 먼저 호출하세요")
        N, L = ids.shape
        out = np.zeros((N, L), dtype=np.float32 if self.mode == "cont" else np.int64)
        if self.mode == "rank":
            return out
        real = ids >= N_SPECIAL
        cols = (ids - N_SPECIAL).clip(min=0)
        rows = np.arange(N)[:, None].repeat(L, axis=1)
        vals = np.asarray(X_clr, float)[rows, cols]
        if self.mode == "bin":
            vals = np.digitize(vals, self.edges_)
        out[real] = vals[real]
        return out

    def fit_transform(self, X_clr, ids):
        return self.fit(X_clr).transform(X_clr, ids)


def lineage_adjacency(lineage: pd.DataFrame, feature_names, level: str = "family",
                      key_col: str | None = None) -> np.ndarray:
    """
    (P, P) 인접행렬. 지정한 분류 단계가 같으면 1.

    level 을 낮출수록(genus) 마스크가 촘촘해지고 높일수록(phylum) 완전연결에
    가까워집니다. 계통 정보가 없는 feature 는 자기 자신하고만 연결됩니다.
    """
    lin = lineage.copy()
    lin.columns = [str(c).lower() for c in lin.columns]
    level = level.lower()
    if level not in lin.columns:
        raise ValueError(f"lineage 에 '{level}' 컬럼이 없습니다 (있는 것: {list(lin.columns)})")
    key = (key_col or ("taxon" if "taxon" in lin.columns else lin.columns[0])).lower()
    lin = lin.drop_duplicates(subset=[key]).set_index(key)

    labels = []
    for i, f in enumerate(feature_names):
        v = lin[level].get(f) if f in lin.index else None
        labels.append(str(v) if v is not None and pd.notna(v) else f"__unknown__{i}")
    labels = np.asarray(labels)
    return (labels[:, None] == labels[None, :]).astype(np.float32)


def attention_mask(ids: np.ndarray, adjacency: np.ndarray,
                   allow_self: bool = True) -> np.ndarray:
    """
    토큰 시퀀스에 맞춘 (N, L, L) 불리언 어텐션 허용 행렬. True = 허용.

    torch 에서 쓰려면:  bias = torch.where(torch.as_tensor(mask), 0.0, -inf)

    주의: 마스크가 너무 촘촘하면 정보가 흐르지 못합니다. 완전히 막힌 행이
    생기면 softmax 가 NaN 이 되므로, 실제 토큰이 아닌 행은 전부 열어 둡니다.
    """
    ids = np.asarray(ids)
    N, L = ids.shape
    A = np.asarray(adjacency)
    real = ids >= N_SPECIAL
    cols = (ids - N_SPECIAL).clip(min=0)
    mask = A[cols[:, :, None], cols[:, None, :]] > 0
    if allow_self:
        mask |= np.eye(L, dtype=bool)[None, :, :]
    mask |= ~real[:, None, :]          # 패딩 열은 막지 않습니다
    mask |= ~real[:, :, None]          # 패딩 행은 통째로 열어 둡니다 (NaN 방지)
    return mask


def sparsity(adjacency: np.ndarray) -> float:
    """인접행렬 밀도. 1.0 이면 완전연결과 같고, 너무 낮으면 정보가 막힙니다."""
    return float(np.asarray(adjacency).mean())
