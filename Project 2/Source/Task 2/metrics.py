import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from dta_data import load_split
from cf import load_cf

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = SCRIPT_DIR
MODEL_DIR = PROJECT_ROOT / 'saved_models'


def calculate_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


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
    if pair_count == 0:
        return 0.0
    return ci / pair_count


def calculate_hit_at_k(recommendations, ground_truth_positives, k=10):
    hits = 0
    total = len(recommendations)
    if total == 0:
        return 0.0
    for uid, recs in recommendations.items():
        top_k = set(recs[:k])
        actual = ground_truth_positives.get(uid, set())
        if top_k.intersection(actual):
            hits += 1
    return hits / total


def evaluate_dataset(dataset_name):
    print("=" * 60)
    print(f"       EVALUATING DATASET: {dataset_name.upper()}       ")
    print("=" * 60)
    
    print(f"1. Loading {dataset_name} dataset splits...")
    train, val, test = load_split(dataset_name)
    print(f"Loaded successfully! Test set shape: {test.shape}")
    
    print(f"\n2. Loading Collaborative Filtering model checkpoint for {dataset_name}...")
    model_path = MODEL_DIR / f"{dataset_name}_cf.npz"
    model = load_cf(model_path)
    print("Model loaded successfully.")

    for df in [train, val, test]:
        df['lig_id'] = df['lig_id'].astype(str).str.strip()
        df['tgt_id'] = df['tgt_id'].astype(str).str.strip()
        
    model.lig_ids = np.array([str(x).strip() for x in model.lig_ids], dtype="U50")
    model.tgt_ids = np.array([str(x).strip() for x in model.tgt_ids], dtype="U50")
    model._lig_pos = {x: i for i, x in enumerate(model.lig_ids)}
    model._tgt_pos = {x: i for i, x in enumerate(model.tgt_ids)}
    
    print("3. Generating predictions on the test set...")
    preds = model.predict_df(test)
    
    lig_col = 'lig_id' if 'lig_id' in test.columns else test.columns[0]
    tgt_col = 'tgt_id' if 'tgt_id' in test.columns else test.columns[1]
    
    potential_names = ['affinity', 'label', 'y', 'score', 'target', 'value']
    target_column = None
    for name in potential_names:
        if name in test.columns:
            target_column = name
            break
    if target_column is None:
        numeric_cols = test.select_dtypes(include=['number']).columns
        target_column = numeric_cols[-1] if len(numeric_cols) > 0 else test.columns[-1]
            
    print(f"Using target column: '{target_column}'")
    
    y_true = test[target_column].values
    y_pred = preds.values if hasattr(preds, 'values') else preds
    
    print("4. Calculating regression metrics (MSE, CI)...")
    mse = calculate_mse(y_true, y_pred)
    ci = calculate_ci(y_true, y_pred)
    
    print(f"\n5. Calculating Recommendation Metrics (Hit@1, Hit@5)...")
    recommendations = {}
    ground_truth_positives = {}
    
    unique_ligands = test[lig_col].unique()
    all_targets = model.tgt_ids
    
    for lig in unique_ligands:
        lig_array = np.full(len(all_targets), lig, dtype="U50")
        preds_all = model.predict(lig_array, all_targets)
        sorted_indices = np.argsort(-preds_all)
        recommendations[lig] = list(all_targets[sorted_indices])
        
        lig_test = test[test[lig_col] == lig]
        threshold = lig_test[target_column].median()
        pos_targets = set(lig_test[lig_test[target_column] >= threshold][tgt_col].astype(str).values)
        if not pos_targets:
            pos_targets = set(lig_test[tgt_col].astype(str).values)
        ground_truth_positives[lig] = pos_targets

    hit_1 = calculate_hit_at_k(recommendations, ground_truth_positives, k=1)
    hit_5 = calculate_hit_at_k(recommendations, ground_truth_positives, k=5)
    
    print("\n" + "-" * 40)
    print(f"       RESULTS ({dataset_name.upper()})       ")
    print("-" * 40)
    print(f" Mean Squared Error (MSE) : {mse:.4f}")
    print(f" Concordance Index (CI)   : {ci:.4f}")
    print(f" Hit@1 Rate               : {hit_1:.4f}")
    print(f" Hit@5 Rate               : {hit_5:.4f}")
    print("-" * 40 + "\n")


def main():
    datasets = ['davis', 'kiba']
    for dataset in datasets:
        evaluate_dataset(dataset)


if __name__ == '__main__':
    main()