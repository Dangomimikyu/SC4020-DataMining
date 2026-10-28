"""
Shared data loading, cleaning and splitting for DAVIS and KIBA (Tasks 2 and 3).

Cleaning mirrors clean_and_dedup() in Task1.ipynb so every task uses the same matrix:
    1. keep observed (non-NaN) and finite scores
    2. DAVIS only: require Kd > 0, then pKd = 9 - log10(Kd)
    3. key ligands by canonical SMILES and targets by amino-acid sequence
    4. average repeated (ligand, target) pairs

Usage:
    from dta_data import load_split
    train, val, test = load_split('davis')   # columns: lig_id, tgt_id, score, ...
"""
import json
import pickle
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / 'Data'
SPLIT_DIR = DATA_DIR / 'splits'
DEEPDTA_URL = 'https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data'
RAW_FILES = ['ligands_can.txt', 'proteins.txt', 'Y']

SEED = 6767                     # same seed as Task1.ipynb
SPLIT_RATIOS = (0.70, 0.15, 0.15)

# score-scale info kept with each dataset (and saved into model checkpoints)
SCALES = {
    'davis': dict(name='pKd', higher_is_stronger=True, limit=5.0),   # Kd = 10,000 nM screening limit
    'kiba':  dict(name='KIBA score', higher_is_stronger=True, limit=None),
}


def download(name):
    """fetch the DeepDTA files into Data/<name>/ if they are not there yet"""
    folder = DATA_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    for f in RAW_FILES:
        if not (folder / f).exists():
            urllib.request.urlretrieve(f'{DEEPDTA_URL}/{name}/{f}', folder / f)
    return folder


class _ArrayUnpickler(pickle.Unpickler):
    """
    the DeepDTA Y files are pickled numpy arrays. a plain pickle.load can run arbitrary
    code from a downloaded file, so only the classes needed to rebuild an array are allowed.
    """
    ALLOWED = {
        ('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'),
        ('numpy', 'ndarray'), ('numpy', 'dtype'),
        ('numpy.core.multiarray', 'scalar'), ('numpy._core.multiarray', 'scalar'),
    }

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f'blocked {module}.{name} in pickle')
        if module.startswith('numpy.core') and int(np.__version__.split('.')[0]) >= 2:
            module = module.replace('numpy.core', 'numpy._core', 1)
        return super().find_class(module, name)


def load_raw(name):
    """
    long table with one row per matrix cell (including missing):
        ligand_cid, target, smiles, sequence, raw_score
    """
    folder = download(name)
    ligands = json.load(open(folder / 'ligands_can.txt'))
    proteins = json.load(open(folder / 'proteins.txt'))
    with open(folder / 'Y', 'rb') as f:
        Y = np.asarray(_ArrayUnpickler(f, encoding='latin1').load(), dtype=float)

    lig_ids, tgt_ids = list(ligands), list(proteins)
    long_df = (pd.DataFrame(Y, index=lig_ids, columns=tgt_ids)
               .rename_axis(index='ligand_cid', columns='target')
               .reset_index()
               .melt(id_vars='ligand_cid', var_name='target', value_name='raw_score'))
    long_df['smiles'] = long_df.ligand_cid.map(ligands)
    long_df['sequence'] = long_df.target.map(proteins)
    return long_df


def canon(smi):
    """canonical SMILES, or the original string if RDKit cannot parse it"""
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
    except ImportError:
        return smi
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else smi


def clean(name):
    """
    cleaned, deduplicated pairs. one row = one (ligand, target) pair
        lig_key  canonical SMILES       lig_id  readable ligand id (first CID / CHEMBL id)
        tgt_key  amino-acid sequence    tgt_id  readable target name (first name)
        score    pKd (DAVIS) or KIBA score, averaged over repeats
        n_repeats
    """
    df = load_raw(name)
    df = df[df.raw_score.notna() & np.isfinite(df.raw_score)]
    if name == 'davis':
        df = df[df.raw_score > 0]
        df = df.assign(score=9 - np.log10(df.raw_score))
    else:
        df = df.assign(score=df.raw_score)

    canon_map = {s: canon(s) for s in df.smiles.unique()}
    df = df.assign(lig_key=df.smiles.map(canon_map), tgt_key=df.sequence)

    # one readable id per key: first name in file order, the same label Task1.ipynb shows
    lig_names = df.groupby('lig_key').ligand_cid.first()
    tgt_names = df.groupby('tgt_key').target.first()

    out = (df.groupby(['lig_key', 'tgt_key'])
             .agg(score=('score', 'mean'), n_repeats=('score', 'size'))
             .reset_index())
    out['lig_id'] = out.lig_key.map(lig_names).astype(str)
    out['tgt_id'] = out.tgt_key.map(tgt_names).astype(str)
    return out[['lig_id', 'tgt_id', 'score', 'n_repeats', 'lig_key', 'tgt_key']]


def make_split(df, ratios=SPLIT_RATIOS, seed=SEED):
    """
    random pair-level split where every ligand and target in val/test also appears in train.

    one random pair per ligand and one per target is forced into train first, the
    remaining pairs are shuffled and cut so the overall sizes match `ratios`.
    each pair appears in exactly one partition.
    """
    # fixed row order by ids from the data files, so the split does not depend on how the
    # installed RDKit version writes canonical SMILES (the groupby order in clean())
    df = df.sort_values(['lig_id', 'tgt_id']).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    n = len(df)
    order = rng.permutation(n)

    # first shuffled occurrence of each ligand / target goes to train
    shuffled = df.iloc[order]
    forced = set(order[~shuffled.lig_id.duplicated().values]) | set(order[~shuffled.tgt_id.duplicated().values])

    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    rest = [i for i in order if i not in forced]
    n_more_train = max(n_train - len(forced), 0)

    split = np.empty(n, dtype=object)
    split[list(forced)] = 'train'
    split[rest[:n_more_train]] = 'train'
    split[rest[n_more_train:n_more_train + n_val]] = 'val'
    split[rest[n_more_train + n_val:]] = 'test'
    return df.assign(split=split)


def check_split(df):
    """assert the split rules from the brief"""
    tr = df[df.split == 'train']
    for part in ('val', 'test'):
        p = df[df.split == part]
        assert p.lig_id.isin(tr.lig_id).all(), f'{part} has ligands not in train'
        assert p.tgt_id.isin(tr.tgt_id).all(), f'{part} has targets not in train'
    assert not df.duplicated(['lig_id', 'tgt_id']).any(), 'a pair appears more than once'


def load_split(name, rebuild=False):
    """
    fixed train/val/test partitions, cached in Data/splits/<name>_split.csv
    so every member and every task uses the identical split.
    """
    path = SPLIT_DIR / f'{name}_split.csv'
    if path.exists() and not rebuild:
        df = pd.read_csv(path, dtype={'lig_id': str, 'tgt_id': str})
    else:
        df = make_split(clean(name))
        SPLIT_DIR.mkdir(parents=True, exist_ok=True)
        df.drop(columns=['lig_key', 'tgt_key']).to_csv(path, index=False)
        df = df.drop(columns=['lig_key', 'tgt_key'])
    check_split(df)
    return tuple(df[df.split == s].reset_index(drop=True) for s in ('train', 'val', 'test'))


if __name__ == '__main__':
    for name in ('davis', 'kiba'):
        tr, va, te = load_split(name, rebuild=True)
        all_ = pd.concat([tr, va, te])
        print(f'{name}: {all_.lig_id.nunique()} ligands, {all_.tgt_id.nunique()} targets, '
              f'{len(all_)} pairs -> train {len(tr)} / val {len(va)} / test {len(te)}')
