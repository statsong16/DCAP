"""
crossmodal.py — 구성 C: 모달리티 간 대조 정렬 (CLIP / COMICAL 방식)

같은 사람의 taxa 프로파일과 pathway 프로파일을 공유 임베딩 공간에서
가깝게, 다른 사람끼리는 멀게 만듭니다. 라벨은 쓰지 않습니다.

여기서 나오는 retrieval accuracy(같은 사람 찾기 정확도)가
"두 모달리티가 실제로 정렬됐는가"의 직접적인 지표입니다.

실행 예:
  python src/crossmodal.py --data data --out runs/crossmodal --epochs 30
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data import MicrobiomeTokenDataset, load_tables
from models import CrossModalAligner, TaxaTransformer
from pretrain_masked import split_by_study


class PairedDataset(torch.utils.data.Dataset):
    """같은 인덱스가 같은 샘플이 되도록 두 모달리티를 묶습니다."""

    def __init__(self, ds_a, ds_b):
        assert len(ds_a) == len(ds_b), "짝이 맞지 않습니다. R 단계에서 교집합을 확인하세요."
        self.a, self.b = ds_a, ds_b

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return self.a[i], self.b[i]


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def run_epoch(model, loader, opt, device, train=True):
    model.train(train)
    tot, acc_sum, n = 0.0, 0.0, 0
    for ba, bb in loader:
        ba, bb = to_device(ba, device), to_device(bb, device)
        if ba["ids"].size(0) < 2:      # 대조학습은 배치에 최소 2개 필요
            continue
        loss, acc, _, _ = model(ba, bb)
        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        bs = ba["ids"].size(0)
        tot += loss.item() * bs
        acc_sum += acc.item() * bs
        n += bs
    return tot / max(n, 1), acc_sum / max(n, 1)


@torch.no_grad()
def export_embeddings(model, ds_a, ds_b, device, batch_size=64):
    """다운스트림에서 쓸 샘플 수준 임베딩(taxa|pathway 결합)을 뽑습니다."""
    model.eval()
    loader = DataLoader(PairedDataset(ds_a, ds_b), batch_size=batch_size, shuffle=False)
    Za, Zb = [], []
    for ba, bb in loader:
        ba, bb = to_device(ba, device), to_device(bb, device)
        za = model.encode(model.enc_a, model.proj_a, ba)
        zb = model.encode(model.enc_b, model.proj_b, bb)
        Za.append(za.cpu().numpy())
        Zb.append(zb.cpu().numpy())
    return np.concatenate(Za), np.concatenate(Zb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/crossmodal")
    ap.add_argument("--value-mode", default="bin", choices=["rank", "bin", "cont"])
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--proj-dim", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    taxa, pathway, meta, _ = load_tables(args.data)
    if pathway is None:
        raise SystemExit("pathway_abundance.csv 가 없습니다. 구성 C 에는 두 모달리티가 필요합니다.")

    ds_a = MicrobiomeTokenDataset(taxa, args.max_len, args.value_mode, args.n_bins)
    ds_b = MicrobiomeTokenDataset(pathway, args.max_len, args.value_mode, args.n_bins)
    tr_idx, va_idx, held = split_by_study(meta, seed=args.seed)
    print(f"학습 {len(tr_idx)} / 검증 {len(va_idx)}  보류 연구: {held or '(샘플 단위 분할)'}")

    mk = lambda ds: TaxaTransformer(ds.vocab_size, d_model=args.d_model,
                                    n_layers=args.n_layers, max_len=args.max_len,
                                    value_mode=args.value_mode, n_bins=args.n_bins)
    model = CrossModalAligner(mk(ds_a), mk(ds_b), proj_dim=args.proj_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    paired = PairedDataset(ds_a, ds_b)
    tr = DataLoader(Subset(paired, tr_idx), batch_size=args.batch_size, shuffle=True)
    va = DataLoader(Subset(paired, va_idx), batch_size=args.batch_size, shuffle=False)

    history, best = [], -1.0
    for ep in range(1, args.epochs + 1):
        trl, tra = run_epoch(model, tr, opt, device, True)
        with torch.no_grad():
            val, vaa = run_epoch(model, va, opt, device, False)
        history.append({"epoch": ep, "train_loss": trl, "train_retrieval": tra,
                        "val_loss": val, "val_retrieval": vaa})
        print(f"epoch {ep:3d} | train {trl:.4f} (retr {tra:.3f}) | val {val:.4f} (retr {vaa:.3f})")
        if vaa > best:
            best = vaa
            torch.save({"model": model.state_dict(), "args": vars(args)},
                       os.path.join(args.out, "aligner.pt"))

    Za, Zb = export_embeddings(model, ds_a, ds_b, device)
    np.savez(os.path.join(args.out, "embeddings.npz"),
             taxa=Za, pathway=Zb, sample_id=ds_a.sample_ids)
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    chance = 1.0 / args.batch_size
    print(f"완료. 최고 검증 retrieval={best:.3f} (배치 내 우연 수준 ≈ {chance:.3f})")
    print(f"임베딩 저장 -> {args.out}/embeddings.npz")


if __name__ == "__main__":
    main()
