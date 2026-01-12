import numpy as np
import seaborn as sns
from itertools import combinations
from tqdm import tqdm


def generate_incoherent_data(adata_in):
    """
    Generates a 'Frankenstein' dataset where Mature RNA profiles are swapped
    within clusters and resampled to match original library size.

    Parameters:
    -----------
    adata_in (AnnData): Annotated data matrix containing physically coherent M 'counts'

    Returns:
    --------
    adata_fake (AnnData): Annotated data matrix containing physically incoherent M 'counts'

    """
    adata_fake = adata_in.copy()

    spliced_bool = adata_in.var['Spliced'].values
    X_spliced = adata_in.layers['counts'][:, spliced_bool].copy()

    if hasattr(X_spliced, "toarray"):
        X_spliced = X_spliced.toarray()

    # preserve library sizes (to trick scVI)
    lib_sizes = X_spliced.sum(axis=1).reshape(-1, 1)
    lib_sizes[lib_sizes == 0] = 1

    P = X_spliced / lib_sizes

    # cluster-restricted shuffling
    P_shuffled = np.zeros_like(P)
    clusters = adata_in.obs['leiden'].unique()

    for clust in clusters:
        indices = np.where(adata_in.obs['leiden'] == clust)[0]
        P_cluster = P[indices, :]

        # Shuffle the rows
        permuted_indices = np.random.permutation(len(indices))
        P_shuffled[indices, :] = P_cluster[permuted_indices, :]

    # resample counts (M')
    M_fake = np.random.poisson(lib_sizes * P_shuffled)

    # inject into new AnnData keeping Nascent (Unspliced) unchanged
    X_new = adata_in.layers['counts'].copy()
    if hasattr(X_new, "toarray"):
         X_new = X_new.toarray()

    X_new[:, spliced_bool] = M_fake

    adata_fake.layers['counts'] = X_new
    adata_fake.X = X_new

    return adata_fake


def plot_ood_separation(nll_real, nll_fake, model_name, ax):
    """
    Plots the KDE distributions of NLL values for real and fake data.

    Parameters:
    -----------
    nll_real, nll_fake (dict): Output dictionaries containing model-derived NLL values for real and fake data.
    model_name (str): Name of model being evaluated (scVI or biVI).
    ax (matplotlib.axes.Axes): Axes object to plot on.

    Returns:
    --------
    None: prints plot to screen

    """
    nll_real_np = nll_real['reconstruction_loss']['reconstruction_loss'].cpu().numpy()
    nll_fake_np = nll_fake['reconstruction_loss']['reconstruction_loss'].cpu().numpy()

    sns.kdeplot(nll_real_np, fill=True, label='Real Data (Coherent)', ax=ax, color='blue')
    sns.kdeplot(nll_fake_np, fill=True, label='Fake Data (Decoupled)', ax=ax, color='red')

    # Calculate "Rejection Score" (Separation between distributions)
    # (Mean_Fake - Mean_Real) / Std_Real
    mu_real = np.mean(nll_real_np)
    mu_fake = np.mean(nll_fake_np)
    std_real = np.std(nll_real_np)

    score = (mu_fake - mu_real) / std_real

    ax.set_title(f"{model_name}\nRejection Score: {score:.2f}")
    ax.set_xlabel("Reconstruction Error (NLL)")
    ax.legend()


def single_pair_kinetic_compensation(adata, bivi_model, cluster_indices=(0, 1), cluster_key='leiden',
                                     lfc_threshold=0.5, mean_threshold=0.1):
    """
    Performs iso-expression analysis between two specified cell clusters to identify
    kinetic compensation using biVI parameters.

    Parameters:
    -----------
    adata (AnnData): Annotated data matrix containing 'counts' in layers and 'Spliced' boolean in var.
    bivi_model (biVI): A trained biVI model capable of returning likelihood parameters.
    cluster_indices (tuple of int) optional: The indices of the cell-type clusters to compare. Default is (0, 1),
        representing the two most frequent clusters.
    cluster_key (str) optional: The key in adata.obs containing cluster assignments. Default is 'leiden'.
    lfc_threshold (float) optional: The log2 fold change cutoff for defining stability and significant parameter
        shifts. Default is 0.5.
    mean_threshold (float) optional: The minimum mean expression required for a gene to be considered 'expressed'.
        Default is 0.1.

    Returns:
    --------
    results (dict): A dictionary containing the total number of genes, indices of iso-expression genes, gene names,
        a boolean mask for compensating genes, calculated log2 fold changes for parameters, and the indices of the
        clusters compared.
    """
    cluster_counts = adata.obs[cluster_key].value_counts()
    cluster_A = cluster_counts.index[cluster_indices[0]]
    cluster_B = cluster_counts.index[cluster_indices[1]]
    print(f"Comparing cell types {cluster_A} and {cluster_B}...")

    idx_A = adata.obs[cluster_key] == cluster_A
    idx_B = adata.obs[cluster_key] == cluster_B

    spliced_bool = adata.var['Spliced'].values
    X_A = adata[idx_A].layers['counts'][:, spliced_bool]
    X_B = adata[idx_B].layers['counts'][:, spliced_bool]

    if hasattr(X_A, "toarray"): X_A = X_A.toarray()
    if hasattr(X_B, "toarray"): X_B = X_B.toarray()

    mean_A = np.mean(X_A, axis=0)
    mean_B = np.mean(X_B, axis=0)
    epsilon = 1e-6
    lfc_mean = np.log2(mean_B + epsilon) - np.log2(mean_A + epsilon)

    mask_expressed = (mean_A > mean_threshold) & (mean_B > mean_threshold)
    mask_stable = (np.abs(lfc_mean) < lfc_threshold) & mask_expressed
    iso_genes_idx = np.where(mask_stable)[0]
    gene_names = np.array(adata.var_names)[spliced_bool]

    print("\nExtracting biVI kinetic parameters...")
    params = bivi_model.get_likelihood_parameters(adata)
    burst_all = params['burst_size']
    deg_all = params['rel_degradation_rate']

    b_A = np.mean(burst_all[idx_A][:, iso_genes_idx], axis=0)
    b_B = np.mean(burst_all[idx_B][:, iso_genes_idx], axis=0)
    gamma_A = np.mean(deg_all[idx_A][:, iso_genes_idx], axis=0)
    gamma_B = np.mean(deg_all[idx_B][:, iso_genes_idx], axis=0)

    lfc_b = np.log2(b_B + epsilon) - np.log2(b_A + epsilon)
    lfc_gamma = np.log2(gamma_B + epsilon) - np.log2(gamma_A + epsilon)

    print("\nQuantifying kinetic compensation...")
    compensating_genes_mask = (np.abs(lfc_b) > lfc_threshold) & \
                               (np.abs(lfc_gamma) > lfc_threshold) & \
                               (np.sign(lfc_b) == np.sign(lfc_gamma))

    results = {
        "total_genes": len(mean_A),
        "iso_genes_idx": iso_genes_idx,
        "gene_names": gene_names,
        "compensating_mask": compensating_genes_mask,
        "lfc_burst": lfc_b,
        "lfc_degradation": lfc_gamma,
        "clusters": (cluster_A, cluster_B)
    }

    return results


def multi_pair_kinetic_compensation(adata, bivi_model, cluster_key='leiden', min_cells=10,
                                    lfc_threshold=0.5, mean_threshold=0.1, min_stable_genes=10):
    """
    Performs pairwise iso-expression analysis across all valid clusters in a dataset to
    identify kinetic compensation patterns.

    Parameters:
    -----------
    adata (AnnData): Annotated data matrix containing 'counts' in layers and 'Spliced' boolean in var.
    bivi_model (biVI): A trained biVI model capable of returning likelihood parameters.
    cluster_key (str) optional: The key in adata.obs containing cluster assignments. Default is 'leiden'.
    min_cells (int) optional: Minimum number of cells required for a cluster to be included. Default is 10.
    lfc_threshold (float) optional: Log2 fold change cutoff for expression stability and parameter shifts. Default is 0.5.
    mean_threshold (float) optional: Minimum mean expression for a gene to be considered expressed. Default is 0.1.
    min_stable_genes (int) optional: Minimum number of stable genes required to process a cluster pair. Default is 10.

    Returns:
    --------
    results (list): A list of dictionaries, each containing metrics and parameter arrays for a specific cluster pair comparison.
    """
    print("Running iso-expression analysis on all cell-type pairs...\n")
    cluster_counts = adata.obs[cluster_key].value_counts()
    valid_clusters = cluster_counts[cluster_counts > min_cells].index.tolist()
    print(f"Analyzing {len(valid_clusters)} valid clusters (>10 cells).")

    pairs = list(combinations(valid_clusters, 2))
    print(f"Processing {len(pairs)} cluster pairs...")

    print("Extracting global biVI parameters...")
    params = bivi_model.get_likelihood_parameters(adata)
    burst_all = params['burst_size']
    deg_all = params['rel_degradation_rate']
    spliced_bool = adata.var['Spliced'].values

    cluster_means = {}
    cluster_params = {}

    for clust in tqdm(valid_clusters, desc="Calculating cluster profiles"):
        idx = adata.obs[cluster_key] == clust
        X_clust = adata[idx].layers['counts'][:, spliced_bool]
        if hasattr(X_clust, "toarray"):
            X_clust = X_clust.toarray()

        cluster_means[clust] = np.mean(X_clust, axis=0)
        cluster_params[clust] = {
            'b': np.mean(burst_all[idx], axis=0),
            'gamma': np.mean(deg_all[idx], axis=0)
        }

    results = []
    epsilon = 1e-6

    for clust_A, clust_B in tqdm(pairs, desc="Analyzing cluster pairs"):
        mean_A = cluster_means[clust_A]
        mean_B = cluster_means[clust_B]
        lfc_mean = np.log2(mean_B + epsilon) - np.log2(mean_A + epsilon)

        mask_expressed = (mean_A > mean_threshold) & (mean_B > mean_threshold)
        mask_stable = (np.abs(lfc_mean) < lfc_threshold) & mask_expressed

        if np.sum(mask_stable) < min_stable_genes:
            continue

        b_A = cluster_params[clust_A]['b'][mask_stable]
        b_B = cluster_params[clust_B]['b'][mask_stable]
        gamma_A = cluster_params[clust_A]['gamma'][mask_stable]
        gamma_B = cluster_params[clust_B]['gamma'][mask_stable]

        lfc_b = np.log2(b_B + epsilon) - np.log2(b_A + epsilon)
        lfc_gamma = np.log2(gamma_B + epsilon) - np.log2(gamma_A + epsilon)

        mask_comp = (np.abs(lfc_b) > lfc_threshold) & \
                    (np.abs(lfc_gamma) > lfc_threshold) & \
                    (np.sign(lfc_b) == np.sign(lfc_gamma))

        n_stable = np.sum(mask_stable)
        n_comp = np.sum(mask_comp)

        results.append({
            'pair': f"{clust_A} vs {clust_B}",
            'n_stable': n_stable,
            'n_compensating': n_comp,
            'pct_compensating': (n_comp / n_stable) * 100,
            'lfc_burst': lfc_b,
            'lfc_gamma': lfc_gamma,
            'mask_comp': mask_comp
        })

    return results

