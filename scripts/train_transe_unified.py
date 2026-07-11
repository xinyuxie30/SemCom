"""
train_transe_unified.py — Train TransE/RotatE on FB15K237/WN18RR/NELL-995.

Supports --model transE (default) or --model rotatE.
Records train+test MRR every --eval-every epochs for convergence analysis.

Usage:
  python scripts/train_transe_unified.py --dataset FB15K237 --model rotatE --epochs 3000
"""
from __future__ import annotations

import argparse, json, random, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

DATA_ROOTS = "C:/Users/xinyuxie/Desktop/iSAC-main/SemanticEncoder/benchmarks"
DATASETS = {"FB15K237": f"{DATA_ROOTS}/FB15K237", "WN18RR": f"{DATA_ROOTS}/WN18RR",
            "NELL-995": f"{DATA_ROOTS}/NELL-995"}

MARGIN, NBATCHES, LR = 5.0, 100, 1.0
DEFAULT_EVAL_MAX, DEFAULT_EVAL_EVERY, DEFAULT_FAST_EVAL_SIZE = 0, 50, 2000


# ═══════════════════════ Data ═══════════════════════

def load_dataset(data_dir):
    base = Path(data_dir); e2id, r2id = {}, {}
    with open(base/"entity2id.txt", encoding="utf-8") as f:
        _ = int(f.readline())
        for l in f:
            p = l.strip().split("\t")
            if len(p) >= 2: e2id[p[0]] = int(p[1])
    with open(base/"relation2id.txt", encoding="utf-8") as f:
        _ = int(f.readline())
        for l in f:
            p = l.strip().split("\t")
            if len(p) >= 2: r2id[p[0]] = int(p[1])
    def _load(fn):
        triples = []
        with open(base/fn, encoding="utf-8") as f:
            _ = int(f.readline())
            for l in f:
                p = l.strip().split()
                if len(p) >= 3:
                    triples.append((int(p[0]), int(p[1]), int(p[2])))
        return triples
    return e2id, r2id, _load("train2id.txt"), _load("valid2id.txt"), _load("test2id.txt")


def build_filter_maps(triples):
    hr2t, rt2h = defaultdict(set), defaultdict(set)
    for h, t, r in triples: hr2t[(h, r)].add(t); rt2h[(r, t)].add(h)
    return hr2t, rt2h


# ═══════════════════════ Models ═══════════════════════

class TransE(nn.Module):
    def __init__(self, ne, nr, dim=128, margin=MARGIN):
        super().__init__()
        self.E = nn.Embedding(ne, dim); self.R = nn.Embedding(nr, dim)
        self.margin = margin
        nn.init.xavier_uniform_(self.E.weight); nn.init.xavier_uniform_(self.R.weight)

    def _score(self, h, r, t):
        h_e = F.normalize(self.E(h), p=2, dim=-1)
        r_e = F.normalize(self.R(r), p=2, dim=-1)
        t_e = F.normalize(self.E(t), p=2, dim=-1)
        return torch.norm(h_e + r_e - t_e, p=1, dim=-1)

    def forward(self, ph, pr, pt, nh, nr, nt):
        return torch.relu(self.margin + self._score(ph, pr, pt) - self._score(nh, nr, nt)).mean()


class RotatE(nn.Module):
    """h∘r≈t in complex space.  h,t∈ℂ^dim, r∈ℝ^dim (rotation angle)."""
    def __init__(self, ne, nr, dim=128, margin=MARGIN):
        super().__init__()
        self.E = nn.Embedding(ne, dim * 2); self.R = nn.Embedding(nr, dim)
        self.margin = margin
        nn.init.xavier_uniform_(self.E.weight); nn.init.xavier_uniform_(self.R.weight)

    def _score(self, h, r, t):
        h_re, h_im = torch.chunk(self.E(h), 2, dim=-1)
        t_re, t_im = torch.chunk(self.E(t), 2, dim=-1)
        r_cos, r_sin = torch.cos(self.R(r)), torch.sin(self.R(r))
        hr_re = h_re * r_cos - h_im * r_sin
        hr_im = h_re * r_sin + h_im * r_cos
        return torch.sqrt((hr_re - t_re)**2 + (hr_im - t_im)**2 + 1e-12).sum(dim=-1)

    def forward(self, ph, pr, pt, nh, nr, nt):
        return torch.relu(self.margin + self._score(ph, pr, pt) - self._score(nh, nr, nt)).mean()


# ═══════════════════════ Sampling ═══════════════════════

class BernoulliSampler:
    def __init__(self, triples, ne):
        self.triples, self.ne = triples, ne
        self.filter_h, self.filter_t = defaultdict(set), defaultdict(set)
        tph, hpt = defaultdict(list), defaultdict(list)
        for h, t, r in triples: self.filter_h[(r, t)].add(h); self.filter_t[(h, r)].add(t)
        for h, t, r in triples:
            tph[r].append(t)
            hpt[r].append(h)
        self.replace_head_prob = {}
        for r in set(tph) | set(hpt):
            tails_per_head = len(tph[r]) / max(1, len(set(hpt[r])))
            heads_per_tail = len(hpt[r]) / max(1, len(set(tph[r])))
            denom = tails_per_head + heads_per_tail
            self.replace_head_prob[r] = tails_per_head / denom if denom > 0 else 0.5

    def sample_batch(self, bs):
        idxs = random.choices(range(len(self.triples)), k=bs)
        phl, prl, ptl, nhl, nrl, ntl = [], [], [], [], [], []
        for i in idxs:
            h, t, r = self.triples[i]; phl.append(h); prl.append(r); ptl.append(t)
            if random.random() < self.replace_head_prob.get(r, 0.5):
                for _ in range(50):
                    c = random.randint(0, self.ne - 1)
                    if c not in self.filter_h[(r, t)]: nhl.append(c); nrl.append(r); ntl.append(t); break
                else: nhl.append(random.randint(0, self.ne - 1)); nrl.append(r); ntl.append(t)
            else:
                for _ in range(50):
                    c = random.randint(0, self.ne - 1)
                    if c not in self.filter_t[(h, r)]: nhl.append(h); nrl.append(r); ntl.append(c); break
                else: nhl.append(h); nrl.append(r); ntl.append(random.randint(0, self.ne - 1))
        return (torch.tensor(phl), torch.tensor(prl), torch.tensor(ptl),
                torch.tensor(nhl), torch.tensor(nrl), torch.tensor(ntl))


# ═══════════════════════ Eval ═══════════════════════

def evaluate_filtered(model, test_triples, all_triples, ne, device, is_rotate=False, eval_max=0):
    model.eval()
    hr2t, rt2h = build_filter_maps(all_triples)
    with torch.no_grad():
        E = model.E.weight.data; R = model.R.weight.data
        if not is_rotate:
            # Match the TransE training score and downstream cosine matching.
            E = F.normalize(E, p=2, dim=-1)
            R = F.normalize(R, p=2, dim=-1)
        else:
            E_re, E_im = torch.chunk(E, 2, dim=-1)
    subset = test_triples if eval_max == 0 else test_triples[:eval_max]
    ranks, t0 = [], time.time()
    for idx, (h, t, r) in enumerate(subset):
        if is_rotate:
            h_re, h_im = torch.chunk(E[h], 2, dim=-1)
            t_re, t_im = torch.chunk(E[t], 2, dim=-1)
            r_cos, r_sin = torch.cos(R[r]).unsqueeze(0), torch.sin(R[r]).unsqueeze(0)
            # tail scores: ||h∘r - all_entities||
            hr_re = h_re * r_cos - h_im * r_sin
            hr_im = h_re * r_sin + h_im * r_cos
            st = torch.sqrt((hr_re - E_re)**2 + (hr_im - E_im)**2 + 1e-12).sum(dim=-1)
            for kt in hr2t.get((h, r), set()):
                if kt != t: st[kt] = float('inf')
            ranks.append(1.0 / ((st < st[t]).sum().item() + 1))
            # head scores: ||all_entities∘r - t||
            eh_re = E_re * r_cos - E_im * r_sin
            eh_im = E_re * r_sin + E_im * r_cos
            sh = torch.sqrt((eh_re - t_re)**2 + (eh_im - t_im)**2 + 1e-12).sum(dim=-1)
            for kh in rt2h.get((r, t), set()):
                if kh != h: sh[kh] = float('inf')
            ranks.append(1.0 / ((sh < sh[h]).sum().item() + 1))
        else:
            h_e = E[h].unsqueeze(0)
            r_e = R[r].unsqueeze(0)
            t_e = E[t].unsqueeze(0)
            st = torch.norm(h_e + r_e - E, p=1, dim=-1)
            for kt in hr2t.get((h, r), set()):
                if kt != t: st[kt] = float('inf')
            ranks.append(1.0 / ((st < st[t]).sum().item() + 1))
            sh = torch.norm(E + r_e - t_e, p=1, dim=-1)
            for kh in rt2h.get((r, t), set()):
                if kh != h: sh[kh] = float('inf')
            ranks.append(1.0 / ((sh < sh[h]).sum().item() + 1))
        if (idx + 1) % 2000 == 0:
            print(f"    filtered: {idx+1}/{len(subset)} [{time.time()-t0:.0f}s]")
    n = len(ranks); mrr = float(np.mean(ranks))
    return {"MRR": round(mrr, 4), "Hits@1": round(sum(1 for r in ranks if r >= 1.0) / n, 4),
            "Hits@3": round(sum(1 for r in ranks if r >= 1.0/3) / n, 4),
            "Hits@10": round(sum(1 for r in ranks if r >= 1.0/10) / n, 4)}


def export_embeddings(model, is_rotate=False):
    """Return checkpoint arrays aligned with evaluation semantics."""
    with torch.no_grad():
        entity = model.E.weight.detach().cpu()
        relation = model.R.weight.detach().cpu()
        if not is_rotate:
            entity = F.normalize(entity, p=2, dim=-1)
            relation = F.normalize(relation, p=2, dim=-1)
        return entity.numpy().astype(np.float32), relation.numpy().astype(np.float32)


# ═══════════════════════ Main ═══════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["FB15K237", "WN18RR", "NELL-995"])
    parser.add_argument("--model", type=str, default="transE", choices=["transE", "rotatE"])
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--nbatches", type=int, default=NBATCHES)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--fast-eval-size", type=int, default=DEFAULT_FAST_EVAL_SIZE)
    parser.add_argument("--eval-max", type=int, default=DEFAULT_EVAL_MAX,
                        help="Limit final evaluation triples; 0 means full test set.")
    parser.add_argument("--save-dir", type=str, default=str(Path(_PROJECT_ROOT) / "checkpoints"))
    parser.add_argument("--tag-suffix", type=str, default="",
                        help="Optional suffix appended to checkpoint tag, e.g. seed42.")
    args = parser.parse_args()

    run_start = time.time()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = args.device; is_rotate = args.model == "rotatE"

    print(f"{args.model} on {args.dataset} | dim={args.dim} | epochs={args.epochs} | seed={args.seed}")
    print("-" * 60)

    e2id, r2id, train, valid, test = load_dataset(DATASETS[args.dataset])
    ne, nr = len(e2id), len(r2id)
    all_triples = train + valid + test
    bs = max(1, len(train) // args.nbatches)
    print(f"  Entities: {ne:,}  Relations: {nr}  Train: {len(train):,}  Test: {len(test):,}")
    print(f"  Batch size: {bs}  (= {len(train)}/{args.nbatches})")

    sampler = BernoulliSampler(train, ne)
    model = (RotatE(ne, nr, args.dim) if is_rotate else TransE(ne, nr, args.dim)).to(device)
    opt = torch.optim.Adagrad(model.parameters(), lr=args.lr)

    history, best_test_mrr = [], 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for _ in range(args.nbatches):
            ph, pr, pt, nh, neg_r, nt = sampler.sample_batch(bs)
            loss = model(ph.to(device), pr.to(device), pt.to(device),
                        nh.to(device), neg_r.to(device), nt.to(device))
            opt.zero_grad(); loss.backward(); opt.step(); total_loss += loss.item()
        avg_loss = total_loss / args.nbatches

        if epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs:
            test_m = evaluate_filtered(model, test[:args.fast_eval_size], all_triples, ne, device, is_rotate)
            train_m = evaluate_filtered(model, train[:args.fast_eval_size], all_triples, ne, device, is_rotate)
            rec = {"epoch": epoch, "loss": round(avg_loss, 6),
                   "train_mrr": train_m["MRR"], "test_mrr": test_m["MRR"],
                   "train_h10": train_m["Hits@10"], "test_h10": test_m["Hits@10"]}
            history.append(rec)
            gap = train_m["MRR"] - test_m["MRR"]
            best_test_mrr = max(best_test_mrr, test_m["MRR"])
            print(f"  Ep {epoch:5d} | loss={avg_loss:.4f} | "
                  f"TRAIN={train_m['MRR']:.4f} | TEST={test_m['MRR']:.4f} | gap={gap:.4f}")

    print("-" * 60)
    print("  Final FULL evaluation...")
    t0 = time.time()
    final = evaluate_filtered(model, test, all_triples, ne, device, is_rotate, eval_max=args.eval_max)
    print(f"  [{time.time()-t0:.0f}s] MRR={final['MRR']:.4f} H@1={final['Hits@1']:.4f} "
          f"H@3={final['Hits@3']:.4f} H@10={final['Hits@10']:.4f}")

    sd = Path(args.save_dir)
    if not sd.is_absolute():
        sd = Path(_PROJECT_ROOT) / sd
    sd.mkdir(parents=True, exist_ok=True)
    tag = f"{args.dataset.lower().replace('-','')}_{args.model.lower()}"
    if args.tag_suffix:
        safe_suffix = args.tag_suffix.strip().replace(" ", "_")
        tag = f"{tag}_{safe_suffix}"
    entity_emb, relation_emb = export_embeddings(model, is_rotate)
    np.save(sd/f"transe_{tag}_entity_emb.npy", entity_emb)
    np.save(sd/f"transe_{tag}_relation_emb.npy", relation_emb)
    with open(sd/f"transe_{tag}_entity2id.json","w") as f: json.dump(e2id, f)
    with open(sd/f"transe_{tag}_relation2id.json","w") as f: json.dump(r2id, f)
    with open(sd/f"transe_{tag}_eval.json","w") as f:
        json.dump({**final, "history": history, "seed": args.seed, "model": args.model,
                   "dim": args.dim, "lr": args.lr, "nbatches": args.nbatches,
                   "normalized_export": not is_rotate,
                   "num_entities": ne, "num_relations": nr,
                   "num_train": len(train), "num_valid": len(valid), "num_test": len(test),
                   "elapsed_sec": round(time.time() - run_start, 2)}, f, indent=2)
    print(f"Saved -> {sd}/transe_{tag}_*  (best_test_MRR={best_test_mrr:.4f})")


if __name__ == "__main__":
    main()
