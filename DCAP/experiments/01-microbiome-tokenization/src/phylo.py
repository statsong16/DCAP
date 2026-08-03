"""
phylo.py — 구성 D: 계통수를 어텐션 제약으로 바꾸는 층.

핵심 발상 (학위 연구의 그래프 사전분포와 같은 것):
  분류군은 독립 항목이 아니라 계통수 위에 놓여 있습니다.
  완전연결 어텐션은 이 구조를 무시하고 L^2 개 쌍을 전부 추정합니다.
  계통상 가까운 분류군끼리만 어텐션을 허용하면
    - 추정할 자유도가 줄고 (표본 부족 보완)
    - 계산량이 줄고
    - 사전지식이 모델에 들어갑니다.

주의: 마스크가 너무 촘촘하면(=거의 다 차단) 정보가 흐르지 못합니다.
      allow_global 로 항상 열어 둘 통로를 남깁니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from data import N_SPECIAL, PAD_ID

RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]


def lineage_matrix(lineage: pd.DataFrame, feature_names: np.ndarray,
                   level: str = "family") -> np.ndarray:
    """
    (P, P) 인접행렬. 지정한 분류 단계가 같으면 1.

    level 을 낮출수록(genus) 마스크가 촘촘해지고,
    높일수록(phylum) 완전연결에 가까워집니다. family 가 보통 무난합니다.
    """
    assert level in RANKS, f"level 은 {RANKS} 중 하나"
    lin = lineage.copy()
    lin.columns = [c.lower() for c in lin.columns]
    key = "taxon" if "taxon" in lin.columns else lin.columns[-1]
    lin = lin.drop_duplicates(subset=[key]).set_index(key)

    labels = []
    for f in feature_names:
        if f in lin.index and level in lin.columns and pd.notna(lin.loc[f, level]):
            labels.append(str(lin.loc[f, level]))
        else:
            labels.append(f"__unknown__{f}")   # 정보 없으면 자기 자신만
    labels = np.asarray(labels)
    return (labels[:, None] == labels[None, :]).astype(np.float32)


def batch_attn_bias(ids: torch.Tensor, adjacency: np.ndarray,
                    allow_global: bool = True) -> torch.Tensor:
    """
    토큰 시퀀스에 맞춰 (B, L, L) 어텐션 편향을 만듭니다.
    허용 = 0.0, 차단 = -inf.

    ids 는 샘플마다 순서가 다르므로 배치마다 인덱싱해서 만들어야 합니다.
    """
    device = ids.device
    B, L = ids.shape
    A = torch.as_tensor(adjacency, device=device)

    is_real = ids >= N_SPECIAL
    idx = (ids - N_SPECIAL).clamp(min=0)                 # (B, L)
    sub = A[idx.reshape(-1)].reshape(B, L, -1)           # (B, L, P)
    sub = torch.gather(sub, 2, idx.unsqueeze(1).expand(B, L, L))  # (B, L, L)

    allowed = sub > 0
    if allow_global:
        eye = torch.eye(L, dtype=torch.bool, device=device).unsqueeze(0)
        allowed = allowed | eye                          # 자기 자신은 항상 허용
        allowed = allowed | (~is_real).unsqueeze(1)      # 패딩 열은 막지 않음
                                                          # (패딩은 key_padding_mask 가 처리)
    # 실제 토큰이 아닌 행은 전부 열어 둡니다 (완전 차단 행 -> softmax nan 방지)
    allowed = allowed | (~is_real).unsqueeze(2)

    bias = torch.zeros(B, L, L, device=device)
    bias.masked_fill_(~allowed, float("-inf"))
    return bias


def sparsity_report(adjacency: np.ndarray) -> str:
    p = adjacency.mean()
    return (f"인접행렬 밀도 {p:.3f}  "
            f"(1.000 이면 완전연결과 동일, 0.05 이하면 너무 촘촘할 수 있음)")
