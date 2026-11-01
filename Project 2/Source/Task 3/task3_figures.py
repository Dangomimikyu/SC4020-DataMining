"""
Task 3 figures, from Results/Task 3/task3_results_<name>.csv (run run_task3.py first)
    fig_t3_overview.png    test MSE and within-ligand CI of every method, per dataset and split
    fig_t3_similarity.png  per cold test target: MSE gain over the CF fallback against the
                           similarity of its nearest training target
    fig_t3_case.png        one DAVIS ligand: true and predicted pKd over its cold test targets
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cold_target import LigandMean, load_cold_split
from run_task3 import OUT_DIR, build
from target_features import similarity

REP_COLOR = {'kmer': '#2a78d6', 'sw': '#eb6834', 'esm': '#1baf7a'}
REP_LABEL = {'kmer': '3-mer TF-IDF', 'sw': 'Smith-Waterman', 'esm': 'ESM-2'}
SPLIT_COLOR = {'random': '#2a78d6', 'cluster': '#eb6834'}
GREY, INK, MUTED = '#9a9893', '#0b0b0b', '#52514e'
TITLE = {'davis': 'DAVIS', 'kiba': 'KIBA'}

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.facecolor': 'white', 'savefig.dpi': 200})


def results():
    return pd.concat(pd.read_csv(OUT_DIR / f'task3_results_{n}.csv') for n in TITLE)


def overview(res):
    test = res[res.part == 'test']
    rows = [('ligand_mean', '-')] + [(m, r) for m in ('seqknn', 'mfmap') for r in REP_COLOR]
    ypos = [0, 1.5, 2.5, 3.5, 5, 6, 7]
    fig, axes = plt.subplots(2, 4, figsize=(11, 4.6), sharey=True)
    for j, (name, split) in enumerate([(n, s) for n in TITLE for s in ('random', 'cluster')]):
        t = test[(test.dataset == name) & (test.split == split)].set_index(['model', 'rep'])
        for i, (metric, label) in enumerate([('mse', 'test MSE (lower is better)'),
                                             ('lig_ci', 'within-ligand CI (higher is better)')]):
            ax = axes[i, j]
            vals = [t.loc[r, metric] for r in rows]
            colors = [GREY] + [REP_COLOR[r] for _, r in rows[1:]]
            ax.barh(ypos, vals, height=0.75, color=colors, edgecolor='white', linewidth=1)
            ax.axvline(vals[0], color=MUTED, lw=0.8, ls='--')
            lo = 0.45 if metric == 'lig_ci' else 0
            ax.set_xlim(lo, max(vals) * 1.08 if metric == 'mse' else 0.9)
            ax.set_xlabel(label)
            ax.grid(axis='x', color='#e6e5e1', lw=0.6)
            ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(f'{TITLE[name]}, {split} cold targets', fontsize=9.5, color=INK)
    axes[0, 0].set_yticks([0, 2.5, 6], ['CF fallback\n(ligand mean)', 'SeqKNN', 'MFMap'])
    axes[0, 0].invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in REP_COLOR.values()]
    fig.tight_layout()
    fig.legend(handles, REP_LABEL.values(), title='target representation', loc='upper center',
               ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(OUT_DIR / 'fig_t3_overview.png', bbox_inches='tight')


def per_target_gain(name, split, model, rep, hp):
    """per test target: MSE of the CF fallback minus MSE of the model, and nearest-train similarity"""
    train, _, test = load_cold_split(name, split)
    test = test.assign(base=LigandMean().fit(train).predict(test),
                       pred=build(name, model, rep, json.loads(hp)).fit(train).predict(test))
    S = similarity(name, 'sw')
    nn = S.loc[test.tgt_id.unique(), train.tgt_id.unique()].max(axis=1)
    g = test.groupby('tgt_id').apply(lambda d: np.mean((d.score - d.base) ** 2) - np.mean((d.score - d.pred) ** 2),
                                     include_groups=False)
    return pd.DataFrame(dict(gain=g, nn_sim=nn.loc[g.index]))


def similarity_plot(res, model='seqknn', rep='sw'):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, name in zip(axes, TITLE):
        for split, c in SPLIT_COLOR.items():
            hp = res[(res.dataset == name) & (res.split == split) & (res.model == model) &
                     (res.rep == rep) & (res.part == 'test')].hp.iloc[0]
            d = per_target_gain(name, split, model, rep, hp)
            ax.scatter(d.nn_sim, d.gain, s=22, color=c, alpha=0.8, edgecolor='white', linewidth=0.6,
                       label=f'{split} cold split')
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel('Smith-Waterman similarity to nearest training target')
        ax.set_title(TITLE[name], fontsize=9.5, color=INK)
        ax.grid(color='#e6e5e1', lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('MSE gain over CF fallback\n(per test target)')
    fig.tight_layout()
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.savefig(OUT_DIR / 'fig_t3_similarity.png', bbox_inches='tight')


def case_plot(res, lig='11338033', model='seqknn', rep='sw', split='random'):
    name = 'davis'
    train, _, test = load_cold_split(name, split)
    hp = res[(res.dataset == name) & (res.split == split) & (res.model == model) &
             (res.rep == rep) & (res.part == 'test')].hp.iloc[0]
    d = test[test.lig_id == lig].copy()
    d['base'] = LigandMean().fit(train).predict(d)
    d['pred'] = build(name, model, rep, json.loads(hp)).fit(train).predict(d)
    d = d.sort_values('score', ascending=False).reset_index(drop=True)
    x = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(x, d.base, color=GREY, lw=2, label='CF fallback (ligand mean)')
    ax.scatter(x, d.score, s=26, color=INK, label='true pKd', zorder=3)
    ax.scatter(x, d.pred, s=26, color=REP_COLOR[rep], label=f'SeqKNN ({REP_LABEL[rep]})', zorder=3)
    top = d.pred.nlargest(5).index
    ax.scatter(x[top], d.pred[top], s=90, facecolor='none', edgecolor=REP_COLOR[rep], lw=1.2,
               label='predicted top 5', zorder=3)
    ax.set_xticks(x, d.tgt_id, rotation=90, fontsize=6)
    ax.set_ylabel('pKd')
    ax.set_title(f'DAVIS ligand {lig}: its {len(d)} cold test targets, sorted by true pKd',
                 fontsize=9.5, color=INK)
    ax.legend(frameon=False, ncol=4, loc='upper right')
    ax.grid(axis='y', color='#e6e5e1', lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_t3_case.png', bbox_inches='tight')


if __name__ == '__main__':
    res = results()
    overview(res)
    similarity_plot(res)
    case_plot(res)
