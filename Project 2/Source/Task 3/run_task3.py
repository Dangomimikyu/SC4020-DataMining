"""
Task 3 experiments: can sequence information rescue CF on cold targets?

for each dataset and cold split ('random' targets, or 'cluster': whole sequence clusters held out)
    baselines  global mean, ligand mean, and Task 2's CF models (MF, item-kNN) refitted on the
               cold training targets, which can only fall back to mu + b_l on a cold target
    proposed   SeqKNN and MFMap on each target representation (kmer, sw, esm)
hyperparameters are chosen on validation MSE, then each chosen model is scored once on test.

    python run_task3.py [davis] [kiba]              tune, then score on test
    python run_task3.py [davis] [kiba] --seeds 5    rescore the chosen models on 5 more splits

writes Results/Task 3/task3_tuning_<name>.csv, task3_results_<name>.csv and task3_seeds_<name>.csv
"""
import itertools
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # shared dta_data.py / cf.py

from cf import MODELS, BiasedMF
from cold_target import GlobalMean, LigandMean, MFMap, SeqKNN, ci, evaluate, load_cold_split, mse
from dta_data import SCALES, SEED
from target_features import REPS, similarity

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / 'Results' / 'Task 3'
SPLITS = ('random', 'cluster')
COLS = ['mse', 'ci', 'lig_ci', 'hit@1', 'hit@5', 'best_rank']
GRIDS = {
    'seqknn': dict(k=[3, 5, 10, 20, 40, 80], gamma=[1, 2, 4, 8, 16], lam=[0.0, 0.01, 0.03, 0.1, 0.3, 1.0]),
    'mfmap':  dict(k=[20, 40, 60], reg=[1.0, 3.0, 5.0], alpha=[0.001, 0.01, 0.1, 1.0], gamma=[1, 2, 4, 8]),
}


def grid(space):
    keys = list(space)
    for vals in itertools.product(*space.values()):
        yield dict(zip(keys, vals))


class Task2CF:
    """a Task 2 CF model, with the hp chosen in Results/task2_tuning_<name>.csv, refitted on cold train"""
    HP_KEYS = {'mf': ['k', 'reg', 'n_iter'], 'knn': ['k', 'shrink', 'reg']}

    def __init__(self, name, kind):
        t = pd.read_csv(ROOT / 'Results' / f'task2_tuning_{name}.csv')
        best = t[t.model == kind].sort_values('val_mse').iloc[0]
        self.hp = {k: int(best[k]) if k in ('k', 'n_iter') else float(best[k]) for k in self.HP_KEYS[kind]}
        self.kind = kind

    def fit(self, train):
        self.m = MODELS[self.kind](**self.hp).fit(train.lig_id, train.tgt_id, train.score)
        return self

    def predict(self, df):
        return self.m.predict_df(df)


def build(name, model, rep, hp, mf=None):
    """an unfitted model, as named in the results table"""
    if model == 'global_mean':
        return GlobalMean()
    if model == 'ligand_mean':
        return LigandMean()
    if model.startswith('task2_'):
        return Task2CF(name, model[len('task2_'):])
    if model == 'seqknn':
        return SeqKNN(similarity(name, rep), **hp)
    return MFMap(similarity(name, rep), mf=mf, **hp)


def tune(name, train, val):
    """grid search on validation; only MSE and CI here, the ranking metrics are for the chosen models"""
    rows, best, mf_cache = [], {}, {}
    for model, space in GRIDS.items():
        for rep in REPS:
            for hp in grid(space):
                mf = None
                if model == 'mfmap':
                    key = (hp['k'], hp['reg'])
                    if key not in mf_cache:
                        mf_cache[key] = BiasedMF(k=hp['k'], reg=hp['reg'], n_iter=15).fit(
                            train.lig_id, train.tgt_id, train.score)
                    mf = mf_cache[key]
                m = build(name, model, rep, hp, mf).fit(train)
                p = m.predict(val)
                rows.append(dict(model=model, rep=rep, **hp, val_mse=mse(val.score, p), val_ci=ci(val.score, p)))
                if (model, rep) not in best or rows[-1]['val_mse'] < best[model, rep][0]:
                    best[model, rep] = (rows[-1]['val_mse'], hp, m)
            print(f'  {model:6s} {rep:4s} best val MSE {best[model, rep][0]:.4f}  {best[model, rep][1]}')
    return pd.DataFrame(rows), best


def run(name):
    limit = SCALES[name]['limit']
    tuning, results = [], []
    for split in SPLITS:
        train, val, test = load_cold_split(name, split)
        print(f'== {name} {split}: targets {train.tgt_id.nunique()}/{val.tgt_id.nunique()}/'
              f'{test.tgt_id.nunique()}, pairs {len(train)}/{len(val)}/{len(test)}')
        t0 = time.time()
        table, best = tune(name, train, val)
        tuning.append(table.assign(split=split))

        chosen = [('global_mean', '-', {}, GlobalMean().fit(train)),
                  ('ligand_mean', '-', {}, LigandMean().fit(train))]
        for kind in ('mf', 'knn'):
            m = Task2CF(name, kind).fit(train)
            chosen.append((f'task2_{kind}', 'id', m.hp, m))
        chosen += [(model, rep, hp, m) for (model, rep), (_, hp, m) in best.items()]

        for model, rep, hp, m in chosen:
            for part, df in (('train', train), ('val', val), ('test', test)):
                results.append(dict(dataset=name, split=split, model=model, rep=rep, hp=json.dumps(hp),
                                    part=part, **evaluate(df, m.predict(df), limit)))
        print(f'   done in {time.time() - t0:.0f}s')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(tuning).to_csv(OUT_DIR / f'task3_tuning_{name}.csv', index=False)
    res = pd.DataFrame(results)
    res.to_csv(OUT_DIR / f'task3_results_{name}.csv', index=False)
    print(res[res.part == 'test'].set_index(['split', 'model', 'rep'])[COLS].round(3).to_string())


def seeds(name, n):
    """
    the chosen models on n more cold splits (seeds SEED+1 .. SEED+n), hyperparameters kept from
    the main split, to see how much the test numbers depend on which targets were held out.
    writes Results/Task 3/task3_seeds_<name>.csv
    """
    limit = SCALES[name]['limit']
    chosen = pd.read_csv(OUT_DIR / f'task3_results_{name}.csv').query("part == 'test'")
    rows = []
    for split in SPLITS:
        for seed in range(SEED + 1, SEED + n + 1):
            train, _, test = load_cold_split(name, split, seed=seed)
            for r in chosen[chosen.split == split].itertuples():
                m = build(name, r.model, r.rep, json.loads(r.hp)).fit(train)
                rows.append(dict(dataset=name, split=split, seed=seed, model=r.model, rep=r.rep,
                                 **evaluate(test, m.predict(test), limit)))
            print(f'  {name} {split} seed {seed} done')
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / f'task3_seeds_{name}.csv', index=False)
    print(out.groupby(['split', 'model', 'rep'], sort=False)[COLS].agg(['mean', 'std']).round(3).to_string())


if __name__ == '__main__':
    args = sys.argv[1:]
    n_seeds = int(args.pop(args.index('--seeds') + 1)) if '--seeds' in args else 0
    names = [a for a in args if a != '--seeds'] or ['davis', 'kiba']
    for name in names:
        if n_seeds:
            seeds(name, n_seeds)
        else:
            run(name)
