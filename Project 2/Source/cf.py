import json

import numpy as np


class _Base:
    kind = None

    def __init__(self, scale=None, **hp):
        self.hp = hp
        self.scale = scale or {}

    # ---------- index mapping ----------
    def _index(self, lig_ids, tgt_ids, scores):
        self.lig_ids = np.array(sorted(set(lig_ids)), dtype=str)
        self.tgt_ids = np.array(sorted(set(tgt_ids)), dtype=str)
        self._lig_pos = {x: i for i, x in enumerate(self.lig_ids)}
        self._tgt_pos = {x: i for i, x in enumerate(self.tgt_ids)}
        li = np.array([self._lig_pos[x] for x in lig_ids])
        ti = np.array([self._tgt_pos[x] for x in tgt_ids])
        return li, ti, np.asarray(scores, dtype=float)

    def _lookup(self, lig_ids, tgt_ids):
        li = np.array([self._lig_pos.get(str(x), -1) for x in lig_ids])
        ti = np.array([self._tgt_pos.get(str(x), -1) for x in tgt_ids])
        return li, ti

    # ---------- shared bias baseline: mu + b_l + b_t ----------
    def _fit_biases(self, li, ti, r, reg, n_iter=10):
        n_l, n_t = len(self.lig_ids), len(self.tgt_ids)
        mu = r.mean()
        bl, bt = np.zeros(n_l), np.zeros(n_t)
        for _ in range(n_iter):
            res = r - mu - bt[ti]
            bl = np.bincount(li, res, n_l) / (np.bincount(li, minlength=n_l) + reg)
            res = r - mu - bl[li]
            bt = np.bincount(ti, res, n_t) / (np.bincount(ti, minlength=n_t) + reg)
        return mu, bl, bt

    def _ligand_means(self, li, r):
        n_l = len(self.lig_ids)
        return np.bincount(li, r, n_l) / np.maximum(np.bincount(li, minlength=n_l), 1)

    def predict_df(self, df):
        return self.predict(df.lig_id.values, df.tgt_id.values)

    # ---------- checkpoint ----------
    def _state(self):
        raise NotImplementedError

    def save(self, path):
        meta = dict(kind=self.kind, hp=self.hp, scale=self.scale)
        np.savez_compressed(path, meta=json.dumps(meta), lig_ids=self.lig_ids,
                            tgt_ids=self.tgt_ids, **self._state())

    def _restore(self, z):
        self.lig_ids, self.tgt_ids = z['lig_ids'].astype(str), z['tgt_ids'].astype(str)
        self._lig_pos = {x: i for i, x in enumerate(self.lig_ids)}
        self._tgt_pos = {x: i for i, x in enumerate(self.tgt_ids)}


class BiasedMF(_Base):
    """
    matrix factorisation with ligand/target biases, fitted by alternating least squares.
    hp: k (latent dim), reg (L2 on factors and biases), n_iter
    """
    kind = 'mf'

    def fit(self, lig_ids, tgt_ids, scores):
        k, reg, n_iter = self.hp.get('k', 10), self.hp.get('reg', 1.0), self.hp.get('n_iter', 20)
        li, ti, r = self._index(lig_ids, tgt_ids, scores)
        n_l, n_t = len(self.lig_ids), len(self.tgt_ids)
        rng = np.random.default_rng(self.hp.get('seed', 0))

        self.mu = r.mean()
        self.lig_mean = self._ligand_means(li, r)
        P = rng.normal(0, 0.1, (n_l, k)); bl = np.zeros(n_l)
        Q = rng.normal(0, 0.1, (n_t, k)); bt = np.zeros(n_t)

        by_lig = [np.flatnonzero(li == u) for u in range(n_l)]
        by_tgt = [np.flatnonzero(ti == t) for t in range(n_t)]
        I = reg * np.eye(k + 1)

        def solve(rows, other_f, other_b, own_idx):
            # each row: least squares for [factors, bias] with features [other factors, 1]
            out_f = np.zeros((len(rows), k)); out_b = np.zeros(len(rows))
            for u, obs in enumerate(rows):
                if len(obs) == 0:
                    continue
                j = own_idx[obs]
                X = np.hstack([other_f[j], np.ones((len(obs), 1))])
                y = r[obs] - self.mu - other_b[j]
                w = np.linalg.solve(X.T @ X + I, X.T @ y)
                out_f[u], out_b[u] = w[:k], w[k]
            return out_f, out_b

        for _ in range(n_iter):
            P, bl = solve(by_lig, Q, bt, ti)
            Q, bt = solve(by_tgt, P, bl, li)
        self.P, self.Q, self.bl, self.bt = P, Q, bl, bt
        return self

    def predict(self, lig_ids, tgt_ids):
        li, ti = self._lookup(lig_ids, tgt_ids)
        pred = np.full(len(li), self.mu)
        kl, kt = li >= 0, ti >= 0
        pred[kl] += self.bl[li[kl]]
        pred[kt] += self.bt[ti[kt]]
        both = kl & kt
        pred[both] += np.einsum('ij,ij->i', self.P[li[both]], self.Q[ti[both]])
        self.last_fallback = dict(unknown_ligand=int((~kl).sum()), unknown_target=int((~kt).sum()))
        return pred

    def _state(self):
        return dict(mu=self.mu, lig_mean=self.lig_mean, P=self.P, Q=self.Q, bl=self.bl, bt=self.bt)

    def _restore(self, z):
        super()._restore(z)
        self.mu = float(z['mu']); self.lig_mean = z['lig_mean']
        self.P, self.Q, self.bl, self.bt = z['P'], z['Q'], z['bl'], z['bt']


class ItemKNN(_Base):
    """
    target-based kNN on baseline residuals (Koren's kNN-baseline).

    similarity between targets = shrunk Pearson-style cosine of residuals over ligands
    measured on both: s_ij * n_ij / (n_ij + shrink).
    prediction for (l, t): baseline + sum_j s_tj * res_lj / sum_j |s_tj| over the k most
    similar targets j that l has a training measurement for (s_tj > 0 only).
    no usable neighbour -> baseline mu + b_l + b_t.
    hp: k (neighbours), shrink, reg (bias regularisation)
    """
    kind = 'knn'

    def fit(self, lig_ids, tgt_ids, scores):
        shrink, reg = self.hp.get('shrink', 10.0), self.hp.get('reg', 5.0)
        li, ti, r = self._index(lig_ids, tgt_ids, scores)
        n_l, n_t = len(self.lig_ids), len(self.tgt_ids)

        self.lig_mean = self._ligand_means(li, r)
        self.mu, self.bl, self.bt = self._fit_biases(li, ti, r, reg)

        # residual matrix with explicit mask; 0 in R means "missing" only together with M == 0
        R = np.zeros((n_l, n_t)); M = np.zeros((n_l, n_t))
        R[li, ti] = r - self.mu - self.bl[li] - self.bt[ti]
        M[li, ti] = 1.0
        self.R, self.M = R, M

        num = R.T @ R
        sq = (R ** 2).T @ M                       # sum over common ligands of res_i^2
        den = np.sqrt(sq * sq.T) + 1e-12
        n_common = M.T @ M
        # targets with no ligand in common get similarity 0 instead of 0/0
        S = np.divide(num * n_common, den * (n_common + shrink),
                      out=np.zeros_like(num), where=n_common > 0)
        np.fill_diagonal(S, 0.0)
        self.S = S
        return self

    def predict(self, lig_ids, tgt_ids):
        k = self.hp.get('k', 20)
        li, ti = self._lookup(lig_ids, tgt_ids)
        pred = np.full(len(li), self.mu)
        kl, kt = li >= 0, ti >= 0
        pred[kl] += self.bl[li[kl]]
        pred[kt] += self.bt[ti[kt]]

        no_neigh = 0
        for q in np.flatnonzero(kl & kt):
            l, t = li[q], ti[q]
            s = self.S[t] * self.M[l]             # similarity to targets this ligand has measured
            cand = np.flatnonzero(s > 0)
            if len(cand) == 0:
                no_neigh += 1
                continue
            top = cand[np.argsort(-s[cand])[:k]]
            pred[q] += s[top] @ self.R[l, top] / s[top].sum()
        self.last_fallback = dict(unknown_ligand=int((~kl).sum()), unknown_target=int((~kt).sum()),
                                  no_neighbour=no_neigh)
        return pred

    def _state(self):
        return dict(mu=self.mu, lig_mean=self.lig_mean, bl=self.bl, bt=self.bt,
                    R=self.R, M=self.M.astype(bool), S=self.S)

    def _restore(self, z):
        super()._restore(z)
        self.mu = float(z['mu']); self.lig_mean = z['lig_mean']
        self.bl, self.bt, self.R, self.S = z['bl'], z['bt'], z['R'], z['S']
        self.M = z['M'].astype(float)


MODELS = {'mf': BiasedMF, 'knn': ItemKNN}


def load_cf(path):
    """rebuild a saved CF model from its .npz checkpoint"""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['meta']))
        model = MODELS[meta['kind']](scale=meta['scale'], **meta['hp'])
        model._restore(z)
    return model
