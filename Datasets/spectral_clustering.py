import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.sparse.linalg import eigsh
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Reference the exact file requested
from graph_datasets import LOADERS, largest_component, summarise

def find_optimal_k(G, max_k=15):
    """Computes the normalized Laplacian and uses the eigengap heuristic."""
    L = nx.normalized_laplacian_matrix(G).astype(float)
    k_evals = min(max_k + 1, G.number_of_nodes() - 1)
    evals, _ = eigsh(L, k=k_evals, which='SM', tol=1e-2)
    evals = np.sort(evals)
    
    # Eigengap heuristic: find the largest gap between consecutive eigenvalues
    gaps = np.diff(evals)
    optimal_k = np.argmax(gaps) + 1 
    return optimal_k, evals

def plot_network_clusters(ax, G, pos, k, adj_matrix, y, title_prefix):
    """Helper function to plot network graphs in a compact boxy frame."""
    if k < 2:
        preds = np.zeros(G.number_of_nodes(), dtype=int)
    else:
        sc = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='kmeans', random_state=42)
        preds = sc.fit_predict(adj_matrix)
    
    mask = y >= 0 
    ari = adjusted_rand_score(y[mask], preds[mask])
    nmi = normalized_mutual_info_score(y[mask], preds[mask])
    print(f"[{title_prefix}] k={k} | ARI: {ari:.4f} | NMI: {nmi:.4f}")
    
    nx.draw_networkx_edges(G, pos, alpha=0.1, ax=ax)
    
    unique_clusters = np.unique(preds)
    norm = mcolors.Normalize(vmin=0, vmax=max(1, k - 1))
    
    for cluster in unique_clusters:
        nodelist = [node for node, p in zip(G.nodes(), preds) if p == cluster]
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=nodelist, 
            node_color=[cm.viridis(norm(cluster))], 
            node_size=15, 
            alpha=0.7, 
            ax=ax
        )
        
    ax.set_title(f"{title_prefix} (k={k})\nARI: {ari:.4f} | NMI: {nmi:.4f}", fontsize=11, pad=8)
    ax.axis('off')

def analyze_datasets():
    datasets_to_run = ["cora", "email-eu-core", "lfr-easy", "lfr-hard"]
    
    for name in datasets_to_run:
        print(f"\n{'='*40}\nProcessing Dataset: {name.upper()}\n{'='*40}")
        
        # 1. Load Data & Perform EDA
        G_full, y_full = LOADERS[name]()
        print("Initial EDA:")
        print(summarise(name, G_full, y_full))
        
        G, y = largest_component(G_full, y_full)
        print("\nEDA on Largest Connected Component:")
        print(summarise(f"{name} (LCC)", G, y))
        
        true_k = len(np.unique(y[y >= 0]))
        print(f"True number of communities: {true_k}")
        
        # 2. Eigengap Analysis
        print("\nComputing optimal k via Eigengap...")
        k_eigengap, evals = find_optimal_k(G)
        print(f"Optimal k found via eigengap: {k_eigengap}")
        
        adj_matrix = nx.to_numpy_array(G)
        pos = nx.spring_layout(G, seed=42, k=0.15)
        
        # 3. Clustering Metric Sweep Analysis (ARI & NMI)
        print(f"Running clustering sweep across cluster sizes for {name}...")
        max_sweep_k = max(15, true_k + 3) if true_k <= 45 else 45
        k_range = list(range(2, max_sweep_k + 1))
        ari_scores = []
        nmi_scores = []

        for k_val in k_range:
            sc_sweep = SpectralClustering(n_clusters=k_val, affinity='precomputed', assign_labels='kmeans', random_state=42)
            preds_sweep = sc_sweep.fit_predict(adj_matrix)
            
            mask = y >= 0
            ari = adjusted_rand_score(y[mask], preds_sweep[mask])
            nmi = normalized_mutual_info_score(y[mask], preds_sweep[mask])
            
            ari_scores.append(ari)
            nmi_scores.append(nmi)
        
        # ------------------------------------------------------------------
        # FIGURE 1: Standalone Laplacian Eigenvalues Plot
        # ------------------------------------------------------------------
        fig_laplacian = plt.figure(figsize=(6, 5))
        ax1 = fig_laplacian.add_subplot(1, 1, 1)
        ax1.plot(range(1, len(evals)+1), evals, marker='o', color='crimson')
        ax1.set_title(f"{name.upper()} - Laplacian Eigenvalues", fontsize=12, fontweight='bold', pad=10)
        ax1.set_xlabel("Index", fontsize=10)
        ax1.set_ylabel("Eigenvalue", fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.5)
        fig_laplacian.tight_layout(pad=2.0)
        
        # ------------------------------------------------------------------
        # FIGURE 2: Standalone Evaluation Metrics Sweep Line Graph
        # ------------------------------------------------------------------
        fig_metrics = plt.figure(figsize=(6.5, 5))
        ax2 = fig_metrics.add_subplot(1, 1, 1)
        ax2.plot(k_range, ari_scores, marker='o', linewidth=2, color='navy', label='ARI Score')
        ax2.plot(k_range, nmi_scores, marker='s', linewidth=2, color='darkorange', label='NMI Score')
        
        ax2.axvline(x=true_k, color='green', linestyle='--', linewidth=1.5, label=f'True k ({true_k})')
        ax2.axvline(x=k_eigengap, color='orange', linestyle=':', linewidth=1.5, label=f'Eigengap k ({k_eigengap})')
        
        ax2.set_title(f"{name.upper()} - Clustering Performance vs. Clusters (k)", fontsize=12, fontweight='bold', pad=10)
        ax2.set_xlabel("Number of Clusters (k)", fontsize=10)
        ax2.set_ylabel("Score (ARI / NMI)", fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(loc='best', fontsize='small')
        fig_metrics.tight_layout(pad=2.0)
        
        # ------------------------------------------------------------------
        # FIGURE 3: Network Cluster Charts (Streamlined 1x2 Layout)
        # ------------------------------------------------------------------
        fig_clusters = plt.figure(figsize=(12, 6))
        fig_clusters.suptitle(f"{name.upper()} - Cluster Visualizations", fontsize=13, fontweight='bold', y=0.98)
        
        # Subplot 1: True Communities
        ax_c1 = fig_clusters.add_subplot(1, 2, 1)
        plot_network_clusters(ax_c1, G, pos, true_k, adj_matrix, y, "True Communities")
        
        # Subplot 2: Eigengap Optimal
        ax_c2 = fig_clusters.add_subplot(1, 2, 2)
        plot_network_clusters(ax_c2, G, pos, k_eigengap, adj_matrix, y, "Eigengap Optimal")
        
        fig_clusters.tight_layout(pad=2.0)
        fig_clusters.subplots_adjust(top=0.88, wspace=0.15)

    plt.show()

if __name__ == "__main__":
    analyze_datasets()