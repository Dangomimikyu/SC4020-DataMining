"""
sequence-only target representations for cold targets (Task 3).

a cold target has no training scores, so CF has nothing to learn its ID from. its amino-acid
sequence is always available, so every representation here is computed from sequences alone:
    kmer  TF-IDF over amino-acid 3-mers (composition), cosine similarity
    sw    Smith-Waterman local alignment score (BLOSUM62), normalised by the self-scores
    esm   mean-pooled ESM-2 embeddings (protein masked language model), centred cosine
none of them sees an affinity label, so computing them over all targets (held-out ones
included) leaks no scores. similarity matrices are cached in Data/features/.
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # shared dta_data.py / cf.py

from dta_data import DATA_DIR, load_raw

FEAT_DIR = DATA_DIR / 'features'
ESM_MODEL = 'facebook/esm2_t30_150M_UR50D'
REPS = ('kmer', 'sw', 'esm')


def target_sequences(name):
    """tgt_id -> amino-acid sequence. tgt_ids are raw target names, as in dta_data.clean()"""
    raw = load_raw(name)
    return raw.drop_duplicates('target').set_index('target').sequence.rename(index=str)


# ---------- k-mer composition ----------
def kmer_similarity(seqs, k=3):
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(analyzer='char', ngram_range=(k, k), lowercase=False,
                        sublinear_tf=True).fit_transform(seqs)   # rows are unit-norm
    return (X @ X.T).toarray()


# ---------- Smith-Waterman alignment ----------
_aligner = None


def _init_aligner():
    global _aligner
    from Bio import Align
    from Bio.Align import substitution_matrices
    _aligner = Align.PairwiseAligner(mode='local', open_gap_score=-11, extend_gap_score=-1,
                                     substitution_matrix=substitution_matrices.load('BLOSUM62'))


def _sw_row(args):
    i, seqs = args
    return [_aligner.score(seqs[i], seqs[j]) for j in range(i, len(seqs))]


def sw_similarity(seqs):
    """S_ab / sqrt(S_aa * S_bb), so identical sequences score 1 and unrelated ones near 0"""
    n = len(seqs)
    with Pool(os.cpu_count(), initializer=_init_aligner) as pool:
        rows = pool.map(_sw_row, [(i, seqs) for i in range(n)], chunksize=4)
    raw = np.zeros((n, n))
    for i, r in enumerate(rows):
        raw[i, i:] = r
    raw = np.triu(raw) + np.triu(raw, 1).T
    d = np.sqrt(np.diag(raw))
    return raw / np.outer(d, d)


# ---------- ESM-2 protein language model ----------
def esm_embeddings(seqs, model_name=ESM_MODEL, window=1022):
    """
    mean of the last-layer residue embeddings. ESM-2 takes at most 1022 residues, so longer
    sequences are cut into windows and the window means are averaged weighted by length.
    """
    import torch
    from transformers import AutoTokenizer, EsmModel

    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name, add_pooling_layer=False).eval().to(device)
    out = []
    with torch.no_grad():
        for s in seqs:
            chunks = [s[i:i + window] for i in range(0, len(s), window)]
            vecs, lens = [], []
            for c in chunks:
                x = tok(c, return_tensors='pt').to(device)
                h = model(**x).last_hidden_state[0, 1:-1]      # drop <cls> and <eos>
                vecs.append(h.mean(0).float().cpu().numpy()); lens.append(len(c))
            out.append(np.average(vecs, axis=0, weights=lens))
    return np.vstack(out)


def esm_similarity(seqs, model_name=ESM_MODEL):
    E = esm_embeddings(seqs, model_name)
    # raw mean embeddings all point the same way (cosine ~0.9 for any pair); centring first
    # leaves the directions that separate proteins
    E = E - E.mean(0)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    return E @ E.T


# ---------- cache ----------
def similarity(name, rep, rebuild=False):
    """all-target similarity matrix as a DataFrame indexed by tgt_id on both axes"""
    path = FEAT_DIR / f'{name}_{rep}_sim.npz'
    if path.exists() and not rebuild:
        with np.load(path) as z:
            ids, S = z['tgt_ids'].astype(str), z['S']
    else:
        seq = target_sequences(name)
        ids = np.array(sorted(seq.index))
        seqs = seq.loc[ids].tolist()
        S = {'kmer': kmer_similarity, 'sw': sw_similarity, 'esm': esm_similarity}[rep](seqs)
        FEAT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, tgt_ids=ids, S=S)
    return pd.DataFrame(S, index=ids, columns=ids)


if __name__ == '__main__':
    import sys
    reps = sys.argv[1:] or list(REPS)
    for name in ('davis', 'kiba'):
        for rep in reps:
            S = similarity(name, rep, rebuild=True)
            off = S.values[~np.eye(len(S), dtype=bool)]
            print(f'{name} {rep}: {len(S)} targets, off-diagonal similarity '
                  f'median {np.median(off):.3f}, 99th pct {np.quantile(off, .99):.3f}')
