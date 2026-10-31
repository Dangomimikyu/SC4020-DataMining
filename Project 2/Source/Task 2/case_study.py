import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from cf import MODELS, load_cf
from dta_data import SCALES, load_split

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = SCRIPT_DIR  

MODEL_DIR = PROJECT_ROOT / 'saved_models'
RESULT_DIR = PROJECT_ROOT / 'Results'
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CASE_LIGANDS = {
    'davis': ['10127622', '11338033', '44259'],
    'kiba': ['CHEMBL1967538', 'CHEMBL2004544', 'CHEMBL565612']
}

def calculate_ci(y_true, y_pred):
    ci = 0.0
    pair_count = 0
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] > y_true[j]:
                pair_count += 1
                if y_pred[i] > y_pred[j]:
                    ci += 1.0
                elif y_pred[i] == y_pred[j]:
                    ci += 0.5
            elif y_true[i] < y_true[j]:
                pair_count += 1
                if y_pred[i] < y_pred[j]:
                    ci += 1.0
                elif y_pred[i] == y_pred[j]:
                    ci += 0.5
    return ci / pair_count if pair_count > 0 else 0.0

def generate_table_image(dataset_name, results_dict):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('off')
    ax.axis('tight')
    
    table_rows = []
    for idx, (lig, data) in enumerate(results_dict.items()):
        for i, (tgt, score, cls) in enumerate(zip(data['targets'], data['scores'], data['classes'])):
            rank = i + 1
            lig_label = str(lig) if rank == 1 else ""
            table_rows.append([lig_label, rank, tgt, f"{score:.4f}", cls])
            
    columns = ['Ligand ID', 'Rank', 'Target ID', 'Predicted Score', 'Classification']
    
    table = ax.table(cellText=table_rows, colLabels=columns, cellLoc='center', loc='center', 
                     colWidths=[0.22, 0.12, 0.22, 0.22, 0.22])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    
    for key, cell in table.get_celld().items():
        row, col = key
        cell.set_facecolor('#ffffff')
        if row == 0:
            cell.set_facecolor('#1e3d59')
            cell.set_text_props(color='white', weight='bold', fontsize=11, ha='center', va='center')
        else:
            cell.set_text_props(va='center', ha='center')
            if col == 0:
                cell.set_text_props(weight='bold', fontsize=11, va='center', ha='center')
                local_row = (row - 1) % 5
                edges = 'LR'
                if local_row == 0:
                    edges += 'T'
                if local_row == 4:
                    edges += 'B'
                cell.visible_edges = edges
            elif col == 4:
                cell.set_text_props(color='#d9534f', weight='bold', va='center', ha='center')
                
    plt.title(f'Case Study Top-5 Recommendations: {dataset_name.upper()} Dataset', fontsize=15, weight='bold', y=1.05)
    plt.tight_layout()
    
    out_path = RESULT_DIR / f"{dataset_name}_case_study_table.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated and saved: {out_path}")

def analyze_and_plot():
    for dataset_name, test_ligs in CASE_LIGANDS.items():
        print("=" * 80)
        print(f"CASE STUDY ANALYSIS: {dataset_name.upper()} DATASET")
        print("=" * 80)
        
        train, val, test = load_split(dataset_name)
        
        for df in [train, val, test]:
            df['lig_id'] = df['lig_id'].astype(str).str.strip()
            df['tgt_id'] = df['tgt_id'].astype(str).str.strip()
        
        model_path = MODEL_DIR / f"{dataset_name}_cf.npz"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        print(f"Loading model from: {model_path}")
        model = load_cf(model_path)
        
        model.lig_ids = np.array([str(x).strip() for x in model.lig_ids], dtype="U50")
        model.tgt_ids = np.array([str(x).strip() for x in model.tgt_ids], dtype="U50")
        model._lig_pos = {x: i for i, x in enumerate(model.lig_ids)}
        model._tgt_pos = {x: i for i, x in enumerate(model.tgt_ids)}
        
        target_column = 'score' if 'score' in test.columns else test.select_dtypes(include=['number']).columns[-1]
        dataset_results = {}
        
        for lig in test_ligs:
            lig_str = str(lig).strip()
            print(f"\n--- Ligand ID: {lig_str} ---")
            
            if lig_str not in model.lig_ids:
                print(f"  [ERROR] Ligand '{lig_str}' NOT found in model.lig_ids!")
                continue
            else:
                idx = model._lig_pos[lig_str]
                print(f"  [INFO] Found ligand at internal index {idx}, bias bl = {model.bl[idx]:.4f}")
            
            lig_test = test[test['lig_id'] == lig_str]
            if len(lig_test) == 0:
                full_df = pd.concat([train, val, test])
                lig_test = full_df[full_df['lig_id'] == lig_str]
                
            if len(lig_test) == 0:
                print(f"  [Error] Ligand {lig_str} not found in dataset records.")
                continue
            
            true_targets = lig_test['tgt_id'].values
            true_scores = lig_test[target_column].values
            
            all_targets = model.tgt_ids
            lig_array = np.full(len(all_targets), lig_str, dtype="U50")
            preds_all = model.predict(lig_array, all_targets)
            
            sorted_indices = np.argsort(-preds_all)
            ranked_targets = all_targets[sorted_indices]
            ranked_scores = preds_all[sorted_indices]
            
            best_idx = np.argmax(true_scores)
            true_best_target = str(true_targets[best_idx])
            true_best_score = true_scores[best_idx]
            
            match_positions = np.where(ranked_targets == true_best_target)[0]
            predicted_rank = int(match_positions[0]) + 1 if len(match_positions) > 0 else None
            
            if predicted_rank is not None:
                predicted_score_of_best = float(ranked_scores[match_positions[0]])
            else:
                predicted_score_of_best = float(model.predict([lig_str], [true_best_target])[0])
            
            hit_1_status = "correct" if predicted_rank is not None and predicted_rank <= 1 else "wrong"
            hit_5_status = "correct" if predicted_rank is not None and predicted_rank <= 5 else "wrong"
            
            pred_subset = model.predict(lig_test['lig_id'].values.astype(str), lig_test['tgt_id'].values.astype(str))
            lig_mse = mean_squared_error(true_scores, pred_subset)
            lig_ci = calculate_ci(true_scores, pred_subset)
            
            print(f"  * Ground Truth Best Target         : {true_best_target} (True Score: {true_best_score:.4f})")
            print(f"  * Predicted Score of Best Target   : {predicted_score_of_best:.4f}")
            print(f"  * Predicted Rank of Best Target    : #{predicted_rank} (out of {len(all_targets)} targets)")
            print(f"  * Hit@1 Status                     : {hit_1_status}")
            print(f"  * Hit@5 Status                     : {hit_5_status}")
            print(f"  * Ligand Subset MSE                : {lig_mse:.4f}")
            print(f"  * Ligand Subset CI                 : {lig_ci:.4f}")
            
            top5_targets = list(ranked_targets[:5])
            top5_scores = list(ranked_scores[:5])
            top5_classes = ["CORRECT" if t == true_best_target else "wrong" for t in top5_targets]
            
            dataset_results[lig_str] = {
                'targets': top5_targets,
                'scores': top5_scores,
                'classes': top5_classes
            }
            
        generate_table_image(dataset_name, dataset_results)

if __name__ == '__main__':
    analyze_and_plot()