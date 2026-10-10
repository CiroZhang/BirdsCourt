"""Trains the reranker using ONLY the train_set source photos from
train_test_split_v3.json. Any real-photo-derived training record (flip/crop/
perspective augmentation, or a raw real-photo failure example) whose source
photo is in test_set is excluded. Fully synthetic decoy renders are always
included (zero real-photo content, nothing to leak).

Saves to Model/weights/reranker_model.pt + pca_artifacts.npz -- this REPLACES
the previous shipped checkpoint (which was trained on all 881 combined,
including derivatives of the very images used for final evaluation). Going
forward, the test_set in train_test_split_v3.json is permanently held out from
training.
"""
import json
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

random.seed(0)
torch.manual_seed(0)

# NOTE: path depth updated after this script moved from Model/scripts/ to
# Model/legacy_reranker/scripts/ when the reranker was superseded by
# ../../honest_scorer.py -- one more dirname() than the original version.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ORIGINAL_PATH = "/scratch/ciro/real_hard_batch/reranker_traindata_v5.jsonl"
HARD_PATH = "/scratch/ciro/real_hard_batch/traindata_hard_v5.jsonl"
SPLIT_PATH = os.path.join(REPO, "BirdsCourtData", "split", "train_test_split.json")
OUT_MODEL = os.path.join(REPO, "Model", "legacy_reranker", "weights", "reranker_model.pt")
OUT_PCA = os.path.join(REPO, "Model", "legacy_reranker", "weights", "pca_artifacts.npz")

CONTEXT_DIM = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ERR_THRESH = 15.0
N_EPOCHS = 200


def source_of(base):
    """Strips known real-photo-derivation suffixes to recover the source
    photo's annotation base name. Returns None for fully synthetic records
    (decoy_* with no real photo involved at all)."""
    if base.startswith("decoy_"):
        return None  # fully synthetic, no real photo -- always safe
    m = re.sub(r"(_flipLR|_crop|_persp)$", "", base)
    return m


def load_filtered(path, train_sources):
    images, skipped, excluded = [], 0, 0
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        src = source_of(d["base"])
        if src is not None and src not in train_sources:
            excluded += 1
            continue
        cands = d["candidates"]
        if len(cands) < 2 or any("corners" not in c for c in cands) or any("context_feat" not in c for c in cands):
            skipped += 1
            continue
        scores = np.array([c["score"] for c in cands], dtype=np.float32)
        s_mean, s_std = scores.mean(), scores.std() + 1e-6
        feats, labels = [], []
        for i, c in enumerate(cands):
            rank_frac = i / max(1, len(cands) - 1)
            w, h = c["img_w"], c["img_h"]
            corners_norm = []
            for (cx, cy) in c["corners"]:
                corners_norm.extend([cx / w, cy / h])
            feats.append([
                (c["score"] - s_mean) / s_std,
                rank_frac,
                c.get("template_alignment", 0.0),
            ] + corners_norm)
            labels.append(1.0 if c["err"] < ERR_THRESH else 0.0)
        raw_context = np.array([c["context_feat"] for c in cands], dtype=np.float32)
        images.append({"base": d["base"], "feats": np.array(feats, dtype=np.float32),
                       "labels": np.array(labels, dtype=np.float32), "errs": [c["err"] for c in cands],
                       "raw_context": raw_context})
    print(f"({path}: {len(images)} kept, {excluded} excluded (test-set source photo), {skipped} skipped (bad data))")
    return images


class SetRanker(nn.Module):
    def __init__(self, n_feat, embed_dim=32):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(n_feat, embed_dim), nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True),
        )
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x):
        h = self.embed(x)
        consensus = h.mean(dim=0, keepdim=True).expand(h.shape[0], -1)
        return self.score_head(torch.cat([h, consensus], dim=1)).squeeze(-1)


def main():
    split = json.load(open(SPLIT_PATH))
    train_sources = set(split["train_set"])
    print(f"{len(train_sources)} train-set source photos, {len(split['test_set'])} held out as test")

    images = load_filtered(ORIGINAL_PATH, train_sources) + load_filtered(HARD_PATH, train_sources)
    print(f"{len(images)} total training images (train-set only, no test-set leakage)")

    pca = PCA(n_components=CONTEXT_DIM, random_state=0)
    train_context = np.concatenate([im["raw_context"] for im in images], axis=0)
    pca.fit(train_context)
    print(f"PCA: {CONTEXT_DIM} components explain {100*pca.explained_variance_ratio_.sum():.1f}% of variance")

    for im in images:
        reduced = pca.transform(im["raw_context"]).astype(np.float32)
        im["feats"] = np.concatenate([im["feats"], reduced], axis=1)
    feat_stack = np.concatenate([im["feats"] for im in images], axis=0)
    f_mean, f_std = feat_stack.mean(0, keepdims=True), feat_stack.std(0, keepdims=True) + 1e-6
    for im in images:
        im["feats"] = (im["feats"] - f_mean) / f_std

    n_feat = images[0]["feats"].shape[1]
    model = SetRanker(n_feat).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)

    for epoch in range(N_EPOCHS):
        model.train()
        random.shuffle(images)
        total_loss, n_pairs = 0.0, 0
        for img in images:
            feats = torch.from_numpy(img["feats"]).to(DEVICE)
            labels = img["labels"]
            pos_idx = np.where(labels > 0.5)[0]
            neg_idx = np.where(labels < 0.5)[0]
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue
            scores = model(feats)
            diff = scores[pos_idx].unsqueeze(1) - scores[neg_idx].unsqueeze(0)
            loss = torch.nn.functional.softplus(-diff).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n_pairs += 1
        if (epoch + 1) % 20 == 0:
            print(f"epoch {epoch+1}/{N_EPOCHS}  loss={total_loss/max(1,n_pairs):.4f}", flush=True)

    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    torch.save(model.state_dict(), OUT_MODEL)
    np.savez(OUT_PCA, pca_components=pca.components_, pca_mean=pca.mean_, f_mean=f_mean, f_std=f_std)
    print(f"saved {OUT_MODEL} (n_feat={n_feat}) and {OUT_PCA}")


if __name__ == "__main__":
    main()
