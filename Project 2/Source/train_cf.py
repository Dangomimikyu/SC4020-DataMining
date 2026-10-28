import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cf import MODELS, load_cf
from dta_data import SCALES, load_split

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / 'saved_models'
RESULT_DIR = ROOT / 'Results'

GRIDS = {
    'mf':  dict(k=[5, 10, 20, 40, 60], reg=[1.0, 3.0, 5.0, 10.0], n_iter=[15]),
    'knn': dict(k=[2, 3, 5, 10, 20], shrink=[0.0, 10.0, 50.0], reg=[5.0]),
}


def baselines(train, val):
    """reference rows for the tuning table: no collaborative information used"""
    mu = train.score.mean()
    lig_mean = train.groupby('lig_id').score.mean()
    rows = []
    for label, f in [('global_mean', lambda d: np.full(len(d), mu)),
                     ('ligand_mean', lambda d: d.lig_id.map(lig_mean).fillna(mu).values)]:
        rows.append(dict(model=label, train_mse=mse(train.score, f(train)), val_mse=mse(val.score, f(val))))
    return rows


def mse(y, p):
    return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))


def grid(space):
    keys = list(space)
    for vals in itertools.product(*space.values()):
        yield dict(zip(keys, vals))


def tune(name):
    train, val, _ = load_split(name)
    scale = SCALES[name]
    rows, best = baselines(train, val), {}
    for kind, space in GRIDS.items():
        for hp in grid(space):
            t0 = time.time()
            m = MODELS[kind](scale=scale, **hp).fit(train.lig_id, train.tgt_id, train.score)
            row = dict(model=kind, **hp,
                       train_mse=mse(train.score, m.predict_df(train)),
                       val_mse=mse(val.score, m.predict_df(val)),
                       **{f'val_{k}': v for k, v in m.last_fallback.items()},
                       seconds=round(time.time() - t0, 1))
            rows.append(row)
            print(f'  {name} {kind} {hp} -> train {row["train_mse"]:.4f}  val {row["val_mse"]:.4f}')
            if kind not in best or row['val_mse'] < best[kind][0]:
                best[kind] = (row['val_mse'], m)
    return pd.DataFrame(rows), best


def main(names):
    MODEL_DIR.mkdir(exist_ok=True); RESULT_DIR.mkdir(exist_ok=True)
    for name in names:
        print(f'== {name}')
        table, best = tune(name)
        table.to_csv(RESULT_DIR / f'task2_tuning_{name}.csv', index=False)

        train, val, test = load_split(name)
        for kind, (_, m) in best.items():
            m.save(MODEL_DIR / f'{name}_{kind}.npz')
        kind, (val_mse, model) = min(best.items(), key=lambda kv: kv[1][0])
        path = MODEL_DIR / f'{name}_cf.npz'
        model.save(path)

        # reload check: the checkpoint alone must reproduce the in-memory predictions
        reloaded = load_cf(path)
        for part in (train, val, test):
            assert np.allclose(model.predict_df(part), reloaded.predict_df(part)), 'reload mismatch'

        print(f'   best: {kind} {model.hp}  val MSE {val_mse:.4f} -> {path.name} (reload verified)')


if __name__ == '__main__':
    main(sys.argv[1:] or ['davis', 'kiba'])
