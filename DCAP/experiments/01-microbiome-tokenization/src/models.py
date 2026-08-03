"""
models.py — 트랜스포머 인코더와 세 가지 학습 헤드.

구성 A : 변수 간 어텐션 (taxa x taxa)        -> TaxaTransformer + MaskedTaxaModel
구성 D : 그래프 마스크 어텐션                -> TaxaTransformer(attn_bias=...)
구성 C : 모달리티 간 정렬 (taxa <-> pathway) -> CrossModalAligner

규모는 COMICAL 과 비슷하게 잡았습니다(2층, d=64, head=4).
파운데이션 모델을 새로 만드는 게 아니라 파이프라인이 돈다는 증거가 목적입니다.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import MASK_ID, PAD_ID


class TaxaTransformer(nn.Module):
    """
    토큰 = 분류군(또는 경로) 하나.
    위치 = 그 샘플 안에서의 풍부도 순위.
    값   = bin 임베딩 / 연속값 투영 / 없음.
    """

    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, dim_ff: int = 128, max_len: int = 256,
                 value_mode: str = "bin", n_bins: int = 10, dropout: float = 0.1):
        super().__init__()
        self.value_mode = value_mode
        self.d_model = d_model
        self.n_heads = n_heads

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.rank_emb = nn.Embedding(max_len, d_model)
        if value_mode == "bin":
            self.value_emb = nn.Embedding(n_bins + 2, d_model, padding_idx=0)
        elif value_mode == "cont":
            self.value_proj = nn.Linear(1, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def embed(self, ids, ranks, vals):
        h = self.token_emb(ids) * math.sqrt(self.d_model)
        h = h + self.rank_emb(ranks.clamp(max=self.rank_emb.num_embeddings - 1))
        if self.value_mode == "bin":
            h = h + self.value_emb(vals)
        elif self.value_mode == "cont":
            h = h + self.value_proj(vals.unsqueeze(-1))
        return h

    def forward(self, ids, ranks, vals, pad_mask, attn_bias=None):
        """
        attn_bias: (B, L, L) float. 0 = 허용, -inf = 차단.
                   구성 D(계통수 제약)에서 사용합니다. None 이면 완전연결 어텐션.
        """
        h = self.embed(ids, ranks, vals)
        mask, key_pad = None, pad_mask
        if attn_bias is not None:
            # PyTorch 는 (B*n_heads, L, L) 를 요구하고,
            # mask 와 key_padding_mask 의 dtype 이 같아야 합니다.
            mask = attn_bias.repeat_interleave(self.n_heads, dim=0)
            key_pad = torch.zeros_like(pad_mask, dtype=mask.dtype)
            key_pad = key_pad.masked_fill(pad_mask, float("-inf"))
        h = self.encoder(h, mask=mask, src_key_padding_mask=key_pad)
        return self.norm(h)

    @staticmethod
    def pool(h, pad_mask):
        """패딩을 제외한 평균 풀링 -> 샘플 수준 임베딩."""
        w = (~pad_mask).float().unsqueeze(-1)
        return (h * w).sum(1) / w.sum(1).clamp(min=1.0)


class MaskedTaxaModel(nn.Module):
    """
    구성 A — 마스킹 복원 사전학습.
    일부 위치의 분류군 ID 를 가리고 무엇이었는지 맞힙니다(Geneformer 방식).
    값 모드가 bin 이면 구간도 함께 예측합니다(보조 손실).
    """

    def __init__(self, encoder: TaxaTransformer, vocab_size: int,
                 n_bins: int = 10, predict_value: bool = True):
        super().__init__()
        self.encoder = encoder
        self.taxa_head = nn.Linear(encoder.d_model, vocab_size)
        self.predict_value = predict_value and encoder.value_mode == "bin"
        if self.predict_value:
            self.value_head = nn.Linear(encoder.d_model, n_bins + 2)

    def forward(self, ids, ranks, vals, pad_mask, attn_bias=None):
        h = self.encoder(ids, ranks, vals, pad_mask, attn_bias)
        out = {"taxa_logits": self.taxa_head(h)}
        if self.predict_value:
            out["value_logits"] = self.value_head(h)
        return out


def apply_mask(ids, vals, pad_mask, mask_ratio: float = 0.15,
               value_mode: str = "bin", generator=None):
    """
    BERT 식 마스킹. 패딩이 아닌 위치 중 mask_ratio 만큼을 MASK 로 바꿉니다.
    반환: (가려진 ids, 가려진 vals, 마스크 위치 bool)
    """
    valid = ~pad_mask
    prob = torch.rand(ids.shape, device=ids.device, generator=generator)
    sel = (prob < mask_ratio) & valid
    # 한 샘플에서 하나도 안 걸리면 손실이 nan 이 되므로 최소 1개는 보장
    empty = (~sel.any(dim=1)) & valid.any(dim=1)
    if empty.any():
        first = valid.float().argmax(dim=1)
        sel[empty, first[empty]] = True

    ids_m = ids.clone()
    ids_m[sel] = MASK_ID
    vals_m = vals.clone()
    if value_mode == "bin":
        vals_m[sel] = 0
    elif value_mode == "cont":
        vals_m[sel] = 0.0
    return ids_m, vals_m, sel


class CrossModalAligner(nn.Module):
    """
    구성 C — 모달리티 간 대조 정렬 (CLIP / COMICAL 방식).
    같은 사람의 taxa 임베딩과 pathway 임베딩을 가깝게, 다른 사람은 멀게.
    """

    def __init__(self, enc_a: TaxaTransformer, enc_b: TaxaTransformer,
                 proj_dim: int = 64, init_temp: float = 0.07):
        super().__init__()
        self.enc_a = enc_a
        self.enc_b = enc_b
        self.proj_a = nn.Linear(enc_a.d_model, proj_dim)
        self.proj_b = nn.Linear(enc_b.d_model, proj_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / init_temp)))

    def encode(self, enc, proj, batch):
        h = enc(batch["ids"], batch["ranks"], batch["vals"], batch["pad_mask"])
        z = proj(enc.pool(h, batch["pad_mask"]))
        return F.normalize(z, dim=-1)

    def forward(self, batch_a, batch_b):
        za = self.encode(self.enc_a, self.proj_a, batch_a)
        zb = self.encode(self.enc_b, self.proj_b, batch_b)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * za @ zb.t()
        target = torch.arange(logits.size(0), device=logits.device)
        loss = 0.5 * (F.cross_entropy(logits, target) +
                      F.cross_entropy(logits.t(), target))
        with torch.no_grad():
            acc = (logits.argmax(dim=1) == target).float().mean()
        return loss, acc, za, zb
