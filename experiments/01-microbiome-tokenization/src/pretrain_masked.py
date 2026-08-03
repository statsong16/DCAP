"""
pretrain_masked.py — 구성 A(+ 선택적으로 구성 D)

자기지도 사전학습: 분류군 토큰 일부를 가리고 무엇이었는지 복원합니다.
라벨(질환 상태)을 전혀 쓰지 않습니다. 이게 요점입니다.

실행 예:
  python src/pretrain_masked.py --data data --out runs/masked --epochs 20
  python src/pretrain_masked.py --data data --out runs/masked_phylo --use-phylo --phylo-level family
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data import MicrobiomeTokenDataset, PAD_ID, load_tables
from models import MaskedTaxaModel, TaxaTransformer, apply_mask
from phylo import batch_attn_bias, lineage_matrix, sparsity_report


def split_by_study(meta, val_frac=0.2, seed=0):
    """study 단위로 나눕니다. 같은 연구가 학습·검증에 걸치면 누수입니다."""
    studies = meta["study_name"].astype(str).to_numpy()
    uniq = np.unique(studies)
    if len(uniq) >= 3:
        rng = np.random.default_rng(seed)
        held = rng.choice(uniq, size=max(1, int(round(len(uniq) * val_frac))), replace=False)
        val = np.where(np.isin(studies, held))[0]
        train = np.where(~np.isin(studies, held))[0]
        return train, val, list(held)
    # 연구가 2개 이하면 어쩔 수 없이 샘플 단위로 나눕니다
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(studies))
    cut = int(len(idx) * (1 - val_frac))
    return idx[:cut], idx[cut:], []


def run_epoch(model, loader, opt, device, value_mode, adjacency=None, train=True):
    model.train(train)
    tot, n, hit, cnt = 0.0, 0, 0, 0
    for batch in loader:
        ids = batch["ids"].to(device)
        ranks = batch["ranks"].to(device)
        vals = batch["vals"].to(device)
        pad = batch["pad_mask"].to(device)

        ids_m, vals_m, sel = apply_mask(ids, vals, pad, 0.15, value_mode)
        bias = batch_attn_bias(ids_m, adjacency) if adjacency is not None else None

        out = model(ids_m, ranks, vals_m, pad, bias)
        logits = out["taxa_logits"][sel]
        target = ids[sel]
        loss = F.cross_entropy(logits, target)

        if "value_logits" in out and value_mode == "bin":
            loss = loss + 0.3 * F.cross_entropy(out["value_logits"][sel], vals[sel])

        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        tot += loss.item() * sel.sum().item()
        n += sel.sum().item()
        hit += (logits.argmax(-1) == target).sum().item()
        cnt += target.numel()
    return tot / max(n, 1), hit / max(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/masked")
    ap.add_argument("--value-mode", default="bin", choices=["rank", "bin", "cont"])
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--use-phylo", action="store_true", help="구성 D: 계통수 어텐션 제약")
    ap.add_argument("--phylo-level", default="family")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    taxa, _, meta, lineage = load_tables(args.data)
    ds = MicrobiomeTokenDataset(taxa, max_len=args.max_len,
                                value_mode=args.value_mode, n_bins=args.n_bins)
    tr_idx, va_idx, held = split_by_study(meta, seed=args.seed)
    print(f"학습 {len(tr_idx)} / 검증 {len(va_idx)}  보류 연구: {held or '(샘플 단위 분할)'}")

    adjacency = None
    if args.use_phylo:
        if lineage is None:
            raise SystemExit("taxa_lineage.csv 가 없어 계통수 마스크를 만들 수 없습니다.")
        adjacency = lineage_matrix(lineage, ds.feature_names, level=args.phylo_level)
        print("구성 D 활성화 —", sparsity_report(adjacency))

    enc = TaxaTransformer(ds.vocab_size, d_model=args.d_model, n_heads=args.n_heads,
                          n_layers=args.n_layers, max_len=args.max_len,
                          value_mode=args.value_mode, n_bins=args.n_bins).to(device)
    model = MaskedTaxaModel(enc, ds.vocab_size, n_bins=args.n_bins).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    tr = DataLoader(Subset(ds, tr_idx), batch_size=args.batch_size, shuffle=True, drop_last=False)
    va = DataLoader(Subset(ds, va_idx), batch_size=args.batch_size, shuffle=False)

    history, best = [], float("inf")
    for ep in range(1, args.epochs + 1):
        trl, tra = run_epoch(model, tr, opt, device, args.value_mode, adjacency, True)
        with torch.no_grad():
            val, vaa = run_epoch(model, va, opt, device, args.value_mode, adjacency, False)
        history.append({"epoch": ep, "train_loss": trl, "train_acc": tra,
                        "val_loss": val, "val_acc": vaa})
        print(f"epoch {ep:3d} | train {trl:.4f} (acc {tra:.3f}) | val {val:.4f} (acc {vaa:.3f})")
        if val < best:
            best = val
            torch.save({"encoder": enc.state_dict(), "args": vars(args),
                        "vocab_size": ds.vocab_size}, os.path.join(args.out, "encoder.pt"))

    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"완료. 최적 val_loss={best:.4f} -> {args.out}/encoder.pt")


if __name__ == "__main__":
    main()
