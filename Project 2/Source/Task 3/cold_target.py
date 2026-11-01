"""
Task 3: cold-target generalisation.

cold split  every pair of a held-out target goes to val or test, so those targets have no
            training scores, only their sequences. ligands stay warm (all occur in training).
metrics     MSE, CI, and per-ligand ranking of the held-out targets (within-ligand CI, Hit@k,
            MRR, rank of the true best target).
models      the CF fallback the brief describes (ligand mean); SeqKNN, a target kNN whose
            neighbours come from sequence similarity instead of shared training ligands; and
            MFMap, Task 2's MF with cold-target factors predicted from sequence similarity.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # shared dta_data.py / cf.py

from cf import BiasedMF
from dta_data import SEED, SPLIT_DIR, SPLIT_RATIOS, load_split
from target_features import similarity

CLUSTER_SIM = 0.30      # targets above this Smith-Waterman similarity (average linkage) share a cluster


# ---------- cold-target split ----------
def target_clusters(name, threshold=CLUSTER_SIM):
    """cluster label per target from average-linkage clustering of 1 - Smith-Waterman similarity"""
    S = similarity(name, 'sw')
    D = np.clip(1 - S.values, 0, None)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    labels = fcluster(linkage(squareform(D), 'average'), 1 - threshold, 'distance')
    return pd.Series(labels, index=S.index)


def make_cold_split(df, groups=None, ratios=SPLIT_RATIOS, seed=SEED):
    """
    target-level split: targets (or whole clusters of them, if `groups` maps tgt_id -> cluster)
    are shuffled and dealt into train / val / test until each holds its share of targets.
    val/test pairs whose ligand has no training pair are dropped, so the test is cold in the
    target only (this can only happen on sparse KIBA).
    """
    df = df.sort_values(['lig_id', 'tgt_id']).reset_index(drop=True)
    tgts = np.array(sorted(df.tgt_id.unique()))
    grp = tgts if groups is None else groups.loc[tgts].values
    units = np.unique(grp)
    units = units[np.random.default_rng(seed).permutation(len(units))]

    bounds = np.cumsum(ratios)[:2] * len(tgts)
    part, filled = {}, 0
    for u in units:
        members = tgts[grp == u]
        part.update(dict.fromkeys(members, 'train' if filled < bounds[0] else
                                  'val' if filled < bounds[1] else 'test'))
        filled += len(members)

    df = df.assign(split=df.tgt_id.map(part))
    train_ligs = df.lig_id[df.split == 'train'].unique()
    keep = (df.split == 'train') | df.lig_id.isin(train_ligs)
    return df[keep].reset_index(drop=True)


def check_cold_split(df):
    tr = df[df.split == 'train']
    assert df.groupby('tgt_id').split.nunique().max() == 1, 'a target is in more than one partition'
    for part in ('val', 'test'):
        p = df[df.split == part]
        assert not p.tgt_id.isin(tr.tgt_id).any(), f'{part} has targets with training scores'
        assert p.lig_id.isin(tr.lig_id).all(), f'{part} has ligands not in train'
    assert not df.duplicated(['lig_id', 'tgt_id']).any(), 'a pair appears more than once'


def load_cold_split(name, kind='random', seed=SEED, rebuild=False):
    """
    kind 'random': individual targets held out. kind 'cluster': whole sequence clusters held out,
    so no test target has a close homologue in training. built from the same cleaned pairs as the
    Task 2 split, cached in Data/splits/<name>_cold_<kind>.csv (other seeds: ..._seed<seed>.csv).
    """
    tag = '' if seed == SEED else f'_seed{seed}'
    path = SPLIT_DIR / f'{name}_cold_{kind}{tag}.csv'
    if path.exists() and not rebuild:
        df = pd.read_csv(path, dtype={'lig_id': str, 'tgt_id': str})
    else:
        pairs = pd.concat(load_split(name)).drop(columns='split')
        groups = target_clusters(name) if kind == 'cluster' else None
        df = make_cold_split(pairs, groups, seed=seed)
        SPLIT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    check_cold_split(df)
    return tuple(df[df.split == s].reset_index(drop=True) for s in ('train', 'val', 'test'))


# ---------- metrics ----------
def mse(y, p):
    return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))


def _untied_pairs(x):
    _, c = np.unique(x, return_counts=True)
    return len(x) * (len(x) - 1) / 2 - (c * (c - 1) / 2).sum()


def ci(y, p):
    """
    concordance index from Kendall tau-b, CI = (tau * sqrt(B / A) + 1) / 2, with A and B the
    pairs with different true and different predicted scores (Task 2 tips). a constant
    prediction gives 0.5.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    A, B = _untied_pairs(y), _untied_pairs(p)
    if A == 0:
        return np.nan
    if B == 0:
        return 0.5
    return float((kendalltau(y, p).statistic * np.sqrt(B / A) + 1) / 2)


def ranking(df, pred, ks=(1, 5), limit=None):
    """
    for each ligand, rank its held-out targets by prediction and find the true best target.
    ties in the prediction are broken at random and scored by their expected value, so a
    constant prediction (the CF fallback) scores exactly the random-ranking baseline.
    ligands with fewer than 2 candidates, or with every score at the screening `limit`,
    have no meaningful best target and are skipped.
    """
    rows = []
    for _, g in df.assign(pred=np.asarray(pred)).groupby('lig_id'):
        y, p = g.score.values, g.pred.values
        if len(y) < 2 or np.ptp(y) == 0 or (limit is not None and y.max() <= limit):
            continue
        best = np.flatnonzero(y == y.max())
        b = best[np.argmax(p[best])]
        above, tied = (p > p[b]).sum(), (p == p[b]).sum()
        ranks = above + np.arange(1, tied + 1)          # positions the best target can take
        rows.append(dict(lig_ci=ci(y, p), best_rank=ranks.mean(), mrr=np.mean(1 / ranks),
                         candidates=len(y),
                         **{f'hit@{k}': np.clip((k - above) / tied, 0, 1) for k in ks}))
    r = pd.DataFrame(rows)
    return dict(r.mean(), n_ligands=len(r))


def evaluate(df, pred, limit=None):
    y = df.score.values
    out = dict(mse=mse(y, pred), ci=ci(y, pred))
    if limit is not None:
        # DAVIS: 70% of labels sit at the pKd 5 screening limit and dominate the MSE
        act = y > limit
        out['mse_active'] = mse(y[act], np.asarray(pred)[act])
    return dict(out, **ranking(df, pred, limit=limit))


# ---------- models ----------
class LigandMean:
    """the CF fallback for a target with no training scores: the ligand's mean training score"""
    def fit(self, train):
        self.mu = train.score.mean()
        self.m = train.groupby('lig_id').score.mean()
        return self

    def predict(self, df):
        return df.lig_id.map(self.m).fillna(self.mu).values


class GlobalMean:
    def fit(self, train):
        self.mu = train.score.mean()
        return self

    def predict(self, df):
        return np.full(len(df), self.mu)


class SeqKNN:
    """
    target kNN for cold targets ("guilt by association": similar proteins bind similar ligands).

    item-kNN CF finds a target's neighbours through its own training scores, which a cold
    target does not have. here the neighbours come from sequence similarity s instead:
        r(l, t) = m_l + sum_j w_j (r_lj - m_l) / (sum_j w_j + lam),   w_j = s(t, j) ** gamma
    over the k training targets j most similar to t that ligand l has a training score for.
    m_l is the ligand's training mean, i.e. the plain CF fallback. lam shrinks towards it when
    the neighbours are only weakly similar; with no usable neighbour the prediction is m_l.
    a target never counts as its own neighbour, so training predictions are leave-one-target-out.
    hp: k, gamma, lam
    """
    def __init__(self, sim, **hp):
        self.sim = sim
        self.hp = hp

    def fit(self, train):
        self.mu = train.score.mean()
        self.lig_ids = np.array(sorted(train.lig_id.unique()))
        self.tgt_ids = np.array(sorted(train.tgt_id.unique()))
        self._lig_pos = {x: i for i, x in enumerate(self.lig_ids)}
        li = train.lig_id.map(self._lig_pos).values
        ti = train.tgt_id.map({x: i for i, x in enumerate(self.tgt_ids)}).values

        R = np.zeros((len(self.lig_ids), len(self.tgt_ids)))
        M = np.zeros_like(R, dtype=bool)
        R[li, ti], M[li, ti] = train.score.values, True
        self.m = R.sum(1) / M.sum(1)
        self.res = np.where(M, R - self.m[:, None], 0.0)
        self.M = M
        S = np.clip(self.sim.loc[:, self.tgt_ids], 0, None)
        for t in set(self.tgt_ids) & set(S.index):
            S.loc[t, t] = 0.0
        self.S = S
        return self

    def predict(self, df):
        k, gamma, lam = self.hp.get('k', 10), self.hp.get('gamma', 1.0), self.hp.get('lam', 0.0)
        li = df.lig_id.map(self._lig_pos).values
        known = ~pd.isna(li)
        pred = np.full(len(df), self.mu)
        for t, rows in df[known].groupby('tgt_id').indices.items():
            rows = np.flatnonzero(known)[rows]
            L = li[rows].astype(int)
            W = np.where(self.M[L], self.S.loc[t].values, 0.0)   # neighbours the ligand was measured on
            if k < W.shape[1]:
                kth = -np.partition(-W, k - 1, axis=1)[:, k - 1:k]
                W = np.where(W >= kth, W, 0.0)
            W = W ** gamma
            num, den = (W * self.res[L]).sum(1), W.sum(1) + lam
            pred[rows] = self.m[L] + np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        return pred


class MFMap:
    """
    Task 2's biased MF, with the factors of cold targets predicted from their sequences.

    MF learns q_t and b_t only for targets with training scores. kernel ridge regression on
    the sequence similarity K = S ** gamma maps any target onto those parameters:
        [q_t, b_t] = K(t, J) (K(J, J) + alpha I)^-1 [Q_J, b_J]      J = training targets
    and the prediction is the usual MF one, r(l, t) = mu + b_l + b_t + p_l . q_t. a target
    unlike every training target gets q_t, b_t near 0, i.e. the CF fallback mu + b_l.
    the MF fit is reused across mapping settings, since it does not depend on them.
    hp: k, reg, n_iter (MF), alpha, gamma (mapping)
    """
    def __init__(self, sim, mf=None, **hp):
        self.sim = sim
        self.mf = mf
        self.hp = hp

    def fit(self, train):
        if self.mf is None:
            self.mf = BiasedMF(k=self.hp.get('k', 20), reg=self.hp.get('reg', 5.0),
                               n_iter=self.hp.get('n_iter', 15)).fit(train.lig_id, train.tgt_id, train.score)
        J = self.mf.tgt_ids
        self.K = np.clip(self.sim.loc[:, J].values, 0, None) ** self.hp.get('gamma', 1.0)
        self._row = {t: i for i, t in enumerate(self.sim.index)}
        KJ = self.K[[self._row[t] for t in J]]
        Y = np.hstack([self.mf.Q, self.mf.bt[:, None]])
        self.A = np.linalg.solve(KJ + self.hp.get('alpha', 0.1) * np.eye(len(J)), Y)
        return self

    def predict(self, df):
        mf = self.mf
        li = np.array([mf._lig_pos.get(x, -1) for x in df.lig_id])
        QB = self.K[[self._row[t] for t in df.tgt_id]] @ self.A
        pred = mf.mu + QB[:, -1]
        kl = li >= 0
        pred[kl] += mf.bl[li[kl]] + np.einsum('ij,ij->i', mf.P[li[kl]], QB[kl, :-1])
        return pred
