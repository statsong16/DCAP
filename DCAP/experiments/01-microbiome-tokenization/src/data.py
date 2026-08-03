"""
data.py — 조성 자료(compositional data)를 토큰 시퀀스로 바꾸는 층.

2주차에 읽은 논문들의 값 인코딩 3가지를 모두 구현해 두고 스위치로 고릅니다.
  - "rank"   : Geneformer 식. 풍부도 순위만 남기고 절대값을 버림
  - "bin"    : scGPT 식. CLR 값을 분위수 구간으로 이산화
  - "cont"   : scFoundation 식. CLR 값을 연속값 그대로 투영

조성 자료 처리 순서:
  상대풍부도 -> (pseudocount) -> CLR -> 순위/구간/연속
CLR 로 폐쇄성(합=1)을 먼저 풀지 않으면 가짜 음의 상관이 그대로 모델에 들어갑니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PAD_ID = 0          # 패딩 토큰
MASK_ID = 1         # 마스킹 토큰
N_SPECIAL = 2       # 실제 taxa 는 2번부터 시작


# ---------------------------------------------------------------------
# 1. 조성 자료 변환
# ---------------------------------------------------------------------
def to_relative(X: np.ndarray) -> np.ndarray:
    """행 합을 1로 맞춥니다. cMD 는 합이 100 인 경우가 많습니다."""
    s = X.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return X / s


def clr_transform(X: np.ndarray, pseudo: str | float = "half_min") -> np.ndarray:
    """
    중심 로그비(centered log-ratio) 변환.

    pseudo:
      "half_min" — 0이 아닌 최솟값의 절반 (마이크로바이옴에서 흔히 쓰는 관행)
      float      — 고정 pseudocount

    0을 어떻게 메우느냐가 결과를 좌우합니다. 이 선택을 README 에 명시하세요.
    """
    X = to_relative(np.asarray(X, dtype=np.float64))
    if pseudo == "half_min":
        nz = X[X > 0]
        eps = (nz.min() / 2.0) if nz.size else 1e-6
    else:
        eps = float(pseudo)
    Xp = X + eps
    logX = np.log(Xp)
    return logX - logX.mean(axis=1, keepdims=True)


# ---------------------------------------------------------------------
# 2. 토큰화
# ---------------------------------------------------------------------
def rank_tokenize(X_rel: np.ndarray, max_len: int, min_abundance: float = 0.0):
    """
    Geneformer 식 순위 인코딩.

    각 샘플에서 검출된 taxa 를 풍부도 내림차순으로 정렬해 시퀀스를 만듭니다.
    반환:
      ids   (N, max_len)  taxa 인덱스 + N_SPECIAL, 빈 자리는 PAD_ID
      ranks (N, max_len)  0,1,2,... 순위(위치 인코딩으로 씀)
    """
    N, P = X_rel.shape
    ids = np.full((N, max_len), PAD_ID, dtype=np.int64)
    ranks = np.zeros((N, max_len), dtype=np.int64)
    for i in range(N):
        row = X_rel[i]
        present = np.where(row > min_abundance)[0]
        if present.size == 0:
            continue
        order = present[np.argsort(-row[present])][:max_len]
        ids[i, : order.size] = order + N_SPECIAL
        ranks[i, : order.size] = np.arange(order.size)
    return ids, ranks


def bin_values(C: np.ndarray, ids: np.ndarray, n_bins: int = 10,
               edges: np.ndarray | None = None):
    """
    scGPT 식 값 이산화. ids 순서에 맞춰 CLR 값을 구간 번호로 바꿉니다.
    구간 경계는 학습셋에서만 계산하고(edges 반환) 검증셋에 재사용하세요.
    0번 구간은 PAD 용으로 비워 둡니다.
    """
    N, L = ids.shape
    if edges is None:
        flat = C[C != 0]
        qs = np.linspace(0, 1, n_bins + 1)[1:-1]
        edges = np.quantile(flat, qs)
    out = np.zeros((N, L), dtype=np.int64)
    for i in range(N):
        valid = ids[i] >= N_SPECIAL
        cols = ids[i, valid] - N_SPECIAL
        out[i, valid] = np.digitize(C[i, cols], edges) + 1
    return out, edges


def cont_values(C: np.ndarray, ids: np.ndarray):
    """scFoundation 식. CLR 값을 자르지 않고 그대로 넘깁니다."""
    N, L = ids.shape
    out = np.zeros((N, L), dtype=np.float32)
    for i in range(N):
        valid = ids[i] >= N_SPECIAL
        cols = ids[i, valid] - N_SPECIAL
        out[i, valid] = C[i, cols]
    return out


# ---------------------------------------------------------------------
# 3. Dataset
# ---------------------------------------------------------------------
class MicrobiomeTokenDataset(Dataset):
    """
    한 모달리티(taxa 또는 pathway)를 토큰 시퀀스로 들고 있는 Dataset.

    value_mode:
      "rank" -> 값 임베딩 없음. 순위(위치)만 씀
      "bin"  -> 이산 구간 임베딩
      "cont" -> 연속값 선형 투영
    """

    def __init__(self, abundance: pd.DataFrame, max_len: int = 256,
                 value_mode: str = "bin", n_bins: int = 10,
                 bin_edges: np.ndarray | None = None):
        assert value_mode in {"rank", "bin", "cont"}
        self.sample_ids = abundance.index.to_numpy()
        self.feature_names = abundance.columns.to_numpy()
        self.value_mode = value_mode
        self.max_len = max_len

        X = abundance.to_numpy(dtype=np.float64)
        X_rel = to_relative(X)
        C = clr_transform(X)

        self.ids, self.ranks = rank_tokenize(X_rel, max_len=max_len)
        if value_mode == "bin":
            self.vals, self.bin_edges = bin_values(C, self.ids, n_bins, bin_edges)
            self.vals = self.vals.astype(np.int64)
        elif value_mode == "cont":
            v = cont_values(C, self.ids)
            # 스케일 안정화: 학습이 발산하지 않도록 표준화
            m, s = v[v != 0].mean(), v[v != 0].std() + 1e-8
            self.vals = ((v - m) / s * (self.ids >= N_SPECIAL)).astype(np.float32)
            self.bin_edges = None
        else:
            self.vals = np.zeros_like(self.ids)
            self.bin_edges = None

        self.clr = C.astype(np.float32)   # 베이스라인 비교용 원본 CLR

    @property
    def vocab_size(self) -> int:
        return len(self.feature_names) + N_SPECIAL

    def __len__(self) -> int:
        return self.ids.shape[0]

    def __getitem__(self, i: int):
        item = {
            "ids": torch.from_numpy(self.ids[i]),
            "ranks": torch.from_numpy(self.ranks[i]),
            "pad_mask": torch.from_numpy(self.ids[i] == PAD_ID),
            "index": torch.tensor(i, dtype=torch.long),
        }
        if self.value_mode == "cont":
            item["vals"] = torch.from_numpy(self.vals[i]).float()
        else:
            item["vals"] = torch.from_numpy(self.vals[i]).long()
        return item


def load_tables(data_dir: str):
    """R 스크립트가 만든 CSV 4개를 읽어 정렬된 상태로 돌려줍니다."""
    import os

    taxa = pd.read_csv(os.path.join(data_dir, "taxa_abundance.csv")).set_index("sample_id")
    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv")).set_index("sample_id")
    path_p = os.path.join(data_dir, "pathway_abundance.csv")
    lin_p = os.path.join(data_dir, "taxa_lineage.csv")

    pathway = pd.read_csv(path_p).set_index("sample_id") if os.path.exists(path_p) else None
    lineage = pd.read_csv(lin_p) if os.path.exists(lin_p) else None

    common = taxa.index.intersection(meta.index)
    if pathway is not None:
        common = common.intersection(pathway.index)
    common = common.sort_values()

    taxa = taxa.loc[common]
    meta = meta.loc[common]
    if pathway is not None:
        pathway = pathway.loc[common]
    return taxa, pathway, meta, lineage
