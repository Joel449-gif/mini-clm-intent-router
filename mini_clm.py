#!/usr/bin/env python3
"""
mini_clm.py - a tiny Contrastive-Language-Model-style router.

Idea (inspired by "Contrastive Language Models", Kwok et al., Stanford/NVIDIA, 2026):
  * Treat a customer message as the STATE and a support workflow as the ACTION.
  * Train two small encoders (frozen text backbone + trainable projection head)
    with a bidirectional InfoNCE loss so each state lands near its correct action
    and far from the others.
  * At inference: embed the state ONCE, compare it to CACHED action embeddings,
    pick the highest cosine similarity. Nothing is generated.

This is a small educational re-implementation, NOT the official CLM-8B model.

Usage:
    python mini_clm.py                          # train + evaluate + benchmark
    python mini_clm.py --backbone tfidf         # offline fallback (no downloads)
    python mini_clm.py --demo "I was charged twice"
    python mini_clm.py --demo "my card got declined" \
        --add-action "Customer reports a failed or declined payment"
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "intents.json"
OUT_DIR = ROOT / "artifacts"
EVAL_PARAPHRASE = 3  # 4th paraphrase is never seen in training


# --------------------------------------------------------------------------- #
# Frozen backbones: text -> normalized vector. Never trained.
# --------------------------------------------------------------------------- #
class MiniLMBackbone:
    name = "minilm"

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        self.dim = getattr(self.model, "get_embedding_dimension", self.model.get_sentence_embedding_dimension)()

    def fit(self, texts):  # frozen pretrained model, nothing to fit
        pass

    @torch.no_grad()
    def encode(self, texts):
        e = self.model.encode(
            list(texts), convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False
        )
        return e.float().cpu()


class TfidfBackbone:
    """Offline fallback so the project runs without downloading a model."""

    name = "tfidf"

    def __init__(self, dim=64):
        self.target_dim = dim

    def fit(self, texts):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(texts)
        X = self.vec.transform(texts)
        k = min(self.target_dim, X.shape[1] - 1, X.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=k, random_state=0).fit(X)
        self.dim = k

    def encode(self, texts):
        z = torch.tensor(self.svd.transform(self.vec.transform(list(texts))), dtype=torch.float32)
        return F.normalize(z, dim=-1)


def make_backbone(name):
    return MiniLMBackbone() if name == "minilm" else TfidfBackbone()


# --------------------------------------------------------------------------- #
# The CLM: two trainable projection heads on top of frozen embeddings.
# --------------------------------------------------------------------------- #
class Head(nn.Module):
    def __init__(self, d_in, d_hidden=256, d_out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(d_hidden, d_out)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class MiniCLM(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.state_head = Head(d_in)
        self.action_head = Head(d_in)
        self.log_scale = nn.Parameter(torch.tensor(float(np.log(1 / 0.07))))  # learnable 1/temperature

    @property
    def scale(self):
        return self.log_scale.exp().clamp(max=100.0)


def info_nce(zs, za, labels, scale):
    """Bidirectional InfoNCE. Several messages share the same action, so in-batch
    entries with the SAME intent (but a different row) are masked out as false negatives."""
    logits = scale * zs @ za.T
    same = labels[:, None] == labels[None, :]
    mask = same & ~torch.eye(len(labels), dtype=torch.bool)
    logits = logits.masked_fill(mask, float("-inf"))
    tgt = torch.arange(len(labels))
    return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt))


def train_clm(model, s_emb, labels, action_pool, epochs=150, bs=32, lr=2e-3, wd=1e-2):
    """s_emb: N x d state embeddings; labels: N intent ids;
    action_pool[intent] -> (n_paraphrases x d) training action embeddings."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = len(labels)
    history = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total, steps = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            if len(idx) < 2:
                continue
            lab = labels[idx]
            a = torch.stack(
                [action_pool[l][random.randrange(len(action_pool[l]))] for l in lab.tolist()]
            )
            loss = info_nce(model.state_head(s_emb[idx]), model.action_head(a), lab, model.scale)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1
        history.append(total / max(steps, 1))
    return history


@torch.no_grad()
def score(model, s_emb, cached_action_emb):
    """The whole inference step: cosine similarity vs. CACHED action embeddings."""
    model.eval()
    return model.state_head(s_emb) @ cached_action_emb.T


@torch.no_grad()
def cache_actions(model, a_emb):
    model.eval()
    return model.action_head(a_emb)


def accuracy(sim, true_idx):
    return (sim.argmax(dim=1) == true_idx).float().mean().item()


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data(holdout):
    intents = json.loads(DATA_PATH.read_text())["intents"]
    ids = [i["id"] for i in intents]
    for h in holdout:
        assert h in ids, f"unknown intent {h}; choose from {ids}"
    return intents


# --------------------------------------------------------------------------- #
# Latency benchmark: cached action embeddings vs re-embedding every candidate
# --------------------------------------------------------------------------- #
@torch.no_grad()
def benchmark(backbone, model, queries, action_texts, ns=(8, 64, 256, 1024), reps=5):
    model.eval()
    rows = []
    for n in ns:
        cands = [f"{action_texts[i % len(action_texts)]} (variant {i})" for i in range(n)]
        cached = model.action_head(backbone.encode(cands))  # done once, offline

        t0 = time.perf_counter()
        for q in queries[:reps]:
            zs = model.state_head(backbone.encode([q]))
            _ = (zs @ cached.T).argmax()
        cached_ms = (time.perf_counter() - t0) / reps * 1000

        t0 = time.perf_counter()
        for q in queries[:reps]:
            zs = model.state_head(backbone.encode([q]))
            za = model.action_head(backbone.encode(cands))  # re-embed every candidate
            _ = (zs @ za.T).argmax()
        uncached_ms = (time.perf_counter() - t0) / reps * 1000
        rows.append((n, cached_ms, uncached_ms))
    return rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def write_report(res, bench_rows, backbone_name, holdout, history):
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "results.json").write_text(
        json.dumps({"backbone": backbone_name, "holdout": holdout, **res, "latency_ms": bench_rows}, indent=2)
    )
    lines = [
        f"# Results (backbone: `{backbone_name}`)",
        "",
        f"Held-out intents (never seen in training): `{', '.join(holdout)}`",
        "",
        "| Setting | Zero-shot backbone | Trained mini-CLM |",
        "|---|---|---|",
        f"| Known actions (unseen wording of the action) | {res['known_zs']:.1%} | {res['known_clm']:.1%} |",
        f"| New actions added at inference, no retraining (only the new-intent messages) | {res['new_zs']:.1%} | {res['new_clm']:.1%} |",
        f"| All actions, all test messages | {res['all_zs']:.1%} | {res['all_clm']:.1%} |",
        "",
        "## Latency per query (CPU, ms)",
        "",
        "| Candidates | Cached action embeddings | Re-embed every candidate |",
        "|---|---|---|",
    ]
    for n, c, u in bench_rows:
        lines.append(f"| {n} | {c:.1f} | {u:.1f} ({u / c:.1f}x slower) |")
    lines += [
        "",
        f"Final training loss: {history[-1]:.3f}. Seeds fixed; small toy dataset, so treat numbers as a demo, not a benchmark.",
    ]
    (OUT_DIR / "results.md").write_text("\n".join(lines))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        labels = ["Known\nactions", "New actions\n(no retrain)", "All"]
        zs = [res["known_zs"], res["new_zs"], res["all_zs"]]
        clm = [res["known_clm"], res["new_clm"], res["all_clm"]]
        x = np.arange(3)
        ax[0].bar(x - 0.2, zs, 0.4, label="Zero-shot backbone")
        ax[0].bar(x + 0.2, clm, 0.4, label="Mini-CLM")
        ax[0].set_xticks(x, labels)
        ax[0].set_ylim(0, 1.05)
        ax[0].set_title("Routing accuracy")
        ax[0].legend()
        ns = [r[0] for r in bench_rows]
        ax[1].plot(ns, [r[1] for r in bench_rows], marker="o", label="Cached actions (CLM-style)")
        ax[1].plot(ns, [r[2] for r in bench_rows], marker="o", label="Re-embed every candidate")
        ax[1].set_xscale("log", base=2)
        ax[1].set_xlabel("Number of candidate actions")
        ax[1].set_ylabel("ms per query")
        ax[1].set_title("Latency")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "results.png", dpi=150)
    except Exception as e:  # plotting is optional
        print(f"(skipped plot: {e})")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=["minilm", "tfidf"], default="minilm")
    p.add_argument("--holdout", default="talk_to_human,update_payment_method",
                   help="comma-separated intent ids never seen in training")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--demo", type=str, default=None, help="route one message using the saved model")
    p.add_argument("--add-action", action="append", default=[],
                   help="(demo) add a brand-new action description at inference time")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    holdout = [h for h in args.holdout.split(",") if h]
    intents = load_data(holdout)
    ids = [i["id"] for i in intents]

    backbone = make_backbone(args.backbone)
    corpus = [m for i in intents for m in i["messages"]] + [a for i in intents for a in i["actions"]]
    backbone.fit(corpus)

    # ---- demo mode -------------------------------------------------------- #
    if args.demo:
        ckpt = torch.load(OUT_DIR / f"clm_{args.backbone}.pt")
        model = MiniCLM(backbone.dim)
        model.load_state_dict(ckpt["model"])
        names = ids + [f"NEW: {a}" for a in args.add_action]
        texts = [i["actions"][0] for i in intents] + args.add_action
        cached = cache_actions(model, backbone.encode(texts))
        sim = score(model, backbone.encode([args.demo]), cached)[0]
        probs = F.softmax(model.scale * sim, dim=-1)
        print(f'\nMessage: "{args.demo}"\n')
        for rank, j in enumerate(probs.argsort(descending=True)[:3].tolist(), 1):
            print(f"  {rank}. {names[j]:<45} {probs[j]:.1%}")
        return

    # ---- split ------------------------------------------------------------ #
    known = [i for i, x in enumerate(intents) if x["id"] not in holdout]
    new = [i for i, x in enumerate(intents) if x["id"] in holdout]
    rng = random.Random(args.seed)
    train_msgs, train_lab, test_msgs, test_lab = [], [], [], []
    for i in known:
        msgs = intents[i]["messages"][:]
        rng.shuffle(msgs)
        train_msgs += msgs[:10]
        train_lab += [i] * 10
        test_msgs += msgs[10:]
        test_lab += [i] * 5
    new_msgs = [m for i in new for m in intents[i]["messages"]]
    new_lab = [i for i in new for _ in intents[i]["messages"]]

    # ---- embed once with the FROZEN backbone ------------------------------ #
    s_train = backbone.encode(train_msgs)
    s_test = backbone.encode(test_msgs)
    s_new = backbone.encode(new_msgs)
    action_pool = {i: backbone.encode(intents[i]["actions"][:EVAL_PARAPHRASE]) for i in known}
    a_eval_raw = backbone.encode([x["actions"][EVAL_PARAPHRASE] for x in intents])  # unseen wording

    # ---- train ------------------------------------------------------------ #
    model = MiniCLM(backbone.dim)
    print(f"Training mini-CLM on {len(train_msgs)} state-action pairs "
          f"({len(known)} known intents, backbone={backbone.name}) ...")
    history = train_clm(model, s_train, torch.tensor(train_lab), action_pool, epochs=args.epochs)
    print(f"  loss: {history[0]:.3f} -> {history[-1]:.3f}")

    # ---- evaluate --------------------------------------------------------- #
    # 1) Known actions only
    k_idx = torch.tensor(known)
    k_map = {g: l for l, g in enumerate(known)}
    k_true = torch.tensor([k_map[t] for t in test_lab])
    clm_known = accuracy(score(model, s_test, cache_actions(model, a_eval_raw[k_idx])), k_true)
    zs_known = accuracy(s_test @ a_eval_raw[k_idx].T, k_true)

    # 2) New actions appear at inference time (candidate set = all intents, no retraining)
    all_cached = cache_actions(model, a_eval_raw)
    new_true = torch.tensor(new_lab)
    clm_new = accuracy(score(model, s_new, all_cached), new_true)
    zs_new = accuracy(s_new @ a_eval_raw.T, new_true)

    # 3) Everything together
    s_all = torch.cat([s_test, s_new])
    y_all = torch.tensor(test_lab + new_lab)
    clm_all = accuracy(score(model, s_all, all_cached), y_all)
    zs_all = accuracy(s_all @ a_eval_raw.T, y_all)

    res = dict(known_clm=clm_known, known_zs=zs_known, new_clm=clm_new, new_zs=zs_new,
               all_clm=clm_all, all_zs=zs_all)

    # ---- latency ---------------------------------------------------------- #
    print("Benchmarking latency ...")
    bench_rows = benchmark(backbone, model, test_msgs, [x["actions"][0] for x in intents])

    OUT_DIR.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict()}, OUT_DIR / f"clm_{args.backbone}.pt")
    write_report(res, bench_rows, backbone.name, holdout, history)
    print("\n" + (OUT_DIR / "results.md").read_text())
    print(f"\nSaved to {OUT_DIR}/ (results.md, results.json, results.png, model weights)")


if __name__ == "__main__":
    main()
