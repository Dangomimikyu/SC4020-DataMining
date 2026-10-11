import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Reference the exact file requested
from graph_datasets import LOADERS, largest_component, summarise

def find_optimal_k_eigengap(G, max_k=15):
    """Computes the normalized Laplacian and uses the eigengap heuristic."""
    L = nx.normalized_laplacian_matrix(G).astype(float)
    k_evals = min(max_k + 1, G.number_of_nodes() - 1)
    evals, _ = eigsh(L, k=k_evals, which='SM', tol=1e-2)
    evals = np.sort(evals)
    
    # Eigengap heuristic: find the largest gap between consecutive eigenvalues
    gaps = np.diff(evals)
    optimal_k = np.argmax(gaps) + 1 
    return optimal_k, evals

def get_self_tuning_affinity(adj_matrix, k_neighbor=7):
    """
    Computes the Zelnik-Manor self-tuning affinity matrix using local scaling.
    Calculates distance to the k-th nearest neighbor for each node to act as its local sigma.
    """
    # For graphs, use shortest path or simply inverted adjacency as a basic distance metric
    # Here we use a basic euclidean distance on the adjacency rows as a proxy for structural equivalence
    distances = cdist(adj_matrix, adj_matrix, metric='euclidean')
    
    # Find distance to the k-th nearest neighbor (local sigma)
    # k_neighbor might need to be bounded by the graph size
    k_idx = min(k_neighbor, distances.shape[0] - 1)
    sorted_dists = np.sort(distances, axis=1)
    local_sigmas = sorted_dists[:, k_idx]
    
    # Avoid division by zero
    local_sigmas[local_sigmas == 0] = 1e-5
    
    # Compute affinity matrix: A_{ij} = exp(- d(s_i, s_j)^2 / (sigma_i * sigma_j))
    sigma_matrix = np.outer(local_sigmas, local_sigmas)
    affinity = np.exp(-(distances ** 2) / sigma_matrix)
    np.fill_diagonal(affinity, 0) # Zero out diagonal
    return affinity

def zelnik_manor_optimal_k(affinity_matrix, max_k=15):
    """
    Approximates the Zelnik-Manor rotation cost heuristic to find optimal K.
    Minimizes the cost function related to eigenvector alignment.
    """
    # Compute Normalized Laplacian of the new affinity matrix
    d = np.sum(affinity_matrix, axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_sym = np.eye(affinity_matrix.shape[0]) - D_inv_sqrt @ affinity_matrix @ D_inv_sqrt
    
    k_evals = min(max_k + 1, affinity_matrix.shape[0] - 1)
    evals, evecs = eigsh(L_sym, k=k_evals, which='SM', tol=1e-2)
    
    costs = []
    # Test k values from 2 to max_k
    k_range = list(range(2, k_evals))
    for k in k_range:
        U = evecs[:, :k]
        # Normalize rows of U
        row_norms = np.linalg.norm(U, axis=1, keepdims=True)
        U_norm = np.divide(U, row_norms, out=np.zeros_like(U), where=row_norms!=0)
        
        # Simplified Zelnik-Manor cost: max value per row should be close to 1 if well-aligned
        # A true implementation uses gradient descent for rotation, this is a fast proxy
        max_vals = np.max(np.abs(U_norm), axis=1)
        cost = np.sum((1 - max_vals) ** 2)
        costs.append(cost)
    
    optimal_k = k_range[np.argmin(costs)]
    return optimal_k, affinity_matrix

def plot_network_clusters(ax, G, pos, k, adj_matrix, y, title_prefix, affinity='precomputed'):
    """Helper function to plot network graphs in a compact boxy frame."""
    if k < 2:
        preds = np.zeros(G.number_of_nodes(), dtype=int)
    else:
        sc = SpectralClustering(n_clusters=k, affinity=affinity, assign_labels='kmeans', random_state=42)
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
        
        # 2. Extract Adjacency & Layout
        adj_matrix = nx.to_numpy_array(G)
        pos = nx.spring_layout(G, seed=42, k=0.15)
        
        # 3. Eigengap Analysis
        print("\nComputing optimal k via Eigengap...")
        k_eigengap, evals = find_optimal_k_eigengap(G)
        print(f"Optimal k found via eigengap: {k_eigengap}")
        
        # 4. Zelnik-Manor Analysis
        print("Computing optimal k via Zelnik-Manor Self-Tuning...")
        zm_affinity = get_self_tuning_affinity(adj_matrix)
        k_zm, _ = zelnik_manor_optimal_k(zm_affinity, max_k=max(15, true_k + 5))
        print(f"Optimal k found via Zelnik-Manor: {k_zm}")
        
        # 5. Clustering Metric Sweep Analysis (ARI & NMI)
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
        ax2.axvline(x=k_zm, color='purple', linestyle='-.', linewidth=1.5, label=f'Zelnik-Manor k ({k_zm})')
        
        ax2.set_title(f"{name.upper()} - Clustering Performance vs. Clusters (k)", fontsize=12, fontweight='bold', pad=10)
        ax2.set_xlabel("Number of Clusters (k)", fontsize=10)
        ax2.set_ylabel("Score (ARI / NMI)", fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(loc='best', fontsize='small')
        fig_metrics.tight_layout(pad=2.0)
        
        # ------------------------------------------------------------------
        # FIGURE 3: Network Cluster Charts (Streamlined 1x3 Layout)
        # ------------------------------------------------------------------
        fig_clusters = plt.figure(figsize=(16, 5))
        fig_clusters.suptitle(f"{name.upper()} - Cluster Visualizations", fontsize=13, fontweight='bold', y=1.02)
        
        # Subplot 1: True Communities
        ax_c1 = fig_clusters.add_subplot(1, 3, 1)
        plot_network_clusters(ax_c1, G, pos, true_k, adj_matrix, y, "True Communities")
        
        # Subplot 2: Eigengap Optimal
        ax_c2 = fig_clusters.add_subplot(1, 3, 2)
        plot_network_clusters(ax_c2, G, pos, k_eigengap, adj_matrix, y, "Eigengap Optimal")

        # Subplot 3: Zelnik-Manor Optimal
        ax_c3 = fig_clusters.add_subplot(1, 3, 3)
        plot_network_clusters(ax_c3, G, pos, k_zm, zm_affinity, y, "Zelnik-Manor Optimal")
        
        fig_clusters.tight_layout(pad=2.0)
        fig_clusters.subplots_adjust(top=0.88, wspace=0.15)

    plt.show()

if __name__ == "__main__":
    analyze_datasets()
