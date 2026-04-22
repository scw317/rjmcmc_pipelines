import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from umap import UMAP


class PostProcess:
    """Organize and analyze posterior sampling results.
    
    The sampling results should the form of BayesBay output,
    however, the BayesBay library is not needed explicitly.
    """
    
    def __init__(self, result_df: pd.DataFrame, sort_ref: str, concatenate_chains: bool = True):
        """
        Parameters
        ----------
        result_df : pd.DataFrame
            This should be the return of 'bayesbay.BayesianInversion.get_results'
            or at least the same form of that.
        concatenate_chains : bool, optional
            The same parameter as 'bayesbay.BayesianInversion.get_results'.
            Decide whether samples from all chains in 'result_df' are aggregated or seperated.
            The default is True.

        Returns
        -------
        None.
        """
        if concatenate_chains:
            if "chain" in result_df.columns:
                self.result_df = result_df
            else:
                # Add the meaningless identical chain number (0) column just for pipeline
                chain_col = pd.Series(np.zeros(result_df), name="chain")
                self.result_df = pd.concat((chain_col, result_df))
        else:
            # Numbering chains from indices of the original dataframe
            df_with_id = result_df.copy()
            df_with_id.index.name = "chain"
            # The original indices becomes the new column named "chain".
            df_with_id = df_with_id.reset_index()
            # Explode chain-wise elements for all columns including the "chain" column
            self.result_df = df_with_id.explode(result_df.columns.tolist()).reset_index(drop=True)

        self.get_names()
        self.sort_df(sort_ref)
    
    def get_names(self):
        """Among the dataframe columns, find names of parameters and their spaces and targets."""
        cols = self.result_df.columns
        # Only trans-dimensional spaces have the ".n_dimensions" parameter.
        trans_space_names = [col.removesuffix(".n_dimensions") for col in cols if col.endswith(".n_dimensions")]
        self.trans_cols = {
            space:[col for col in cols if col.startswith(f"{space}.") and not col.endswith(".n_dimensions")]
            for space in trans_space_names
        }
        self.target_cols = [col for col in cols if col.endswith(".dpred")]
        # Anything not matched at the above are fixed-dimensional parameters.
        exclude_set = {"chain"}
        exclude_set.update(self.target_cols)
        exclude_set.update([col for col in cols for space in trans_space_names if col.startswith(f"{space}.")])
        fixed_col_names = list(set(cols) - exclude_set)
        self.fixed_cols = {}
        for col in fixed_col_names:
            space = col.split('.', 1)[0]
            self.fixed_cols.setdefault(space, []).append(col)
    
    def sort_df_by_dim(self, df: pd.DataFrame, sort_ref: str):
        ref_col= np.array(df[sort_ref].tolist())
        indices = np.argsort(ref_col, axis=1)
        sorted_df = pd.DataFrame(index=df.index)
        def wrapper(x):
            return np.take_along_axis(np.array(x.tolist()), indices, axis=1)
        sorted_df[list(self.trans_cols)].apply(wrapper)
        
    def sort_df(self, sort_ref: str):
        self

    def analyze_posterior_modes(self, samples, var_threshold: float = 0.95, init_method="index"):
        """Robust pipeline for high-dimensional posterior analysis.
        
        Handles extremely small sample sizes and non-linear manifolds.
        
        Parameters
        ----------
        samples : np.ndarray
            shape (n_samples, n_params)
        var_threshold : float
            variance ratio for dimensionality reduction.
        init_method : str
            'index' (nearest sample) or 'inverse' (UMAP inverse_transform).
        """
        n_samples, n_params = samples.shape
        
        # Check for absolute minimum samples to prevent failure
        if n_samples < 2:
            raise ValueError(f"Insufficient samples (N={n_samples}). Minimum N=2 required.")
    
        # Data standardization: Handling zero-variance axes to avoid division by zero
        std_val = np.std(samples, axis=0)
        std_val[std_val == 0] = 1.0 
        samples_std = (samples - np.mean(samples, axis=0)) / std_val
    
        # --- Step 1: Dimensionality Selection via PCA ---
        # SVD-based PCA to determine the intrinsic dimensionality of the manifold
        pca = PCA().fit(samples_std)
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        target_dim = np.argmax(cum_var >= var_threshold) + 1
        
        # Logic: Only reduce if target_dim < n_params and N > target_dim
        if target_dim >= n_params:
            print(f"Reduction bypassed: Target dim ({target_dim}) >= Original ({n_params}).")
            reduced_space = samples_std
            use_reduction = False
        else:
            print(f"Target dimension: {target_dim} (Explains {cum_var[target_dim-1]:.2%} variance).")
            use_reduction = True
    
        # --- Step 2: Non-linear Projection via UMAP ---
        reducer = None
        if use_reduction:
            # Adaptive neighbors based on sample size to prevent algorithm collapse
            adj_neighbors = min(15, n_samples - 1)
            try:
                reducer = UMAP(n_components=target_dim, n_neighbors=adj_neighbors, random_state=42)
                reduced_space = reducer.fit_transform(samples_std)
            except Exception as e:
                warnings.warn(f"UMAP transformation failed: {e}. Falling back to PCA space.")
                reduced_space = PCA(n_components=min(target_dim, n_samples-1)).fit_transform(samples_std)
                reducer = None
        else:
            reduced_space = samples_std
    
        # --- Step 3: Mode Detection via HDBSCAN ---
        # Adaptive cluster size based on dataset scale
        adj_min_cluster = max(2, int(n_samples * 0.01)) if n_samples > 10 else 2
        clusterer = HDBSCAN(min_cluster_size=adj_min_cluster)
        labels = clusterer.fit_predict(reduced_space)
        
        unique_labels = [l for l in np.unique(labels) if l != -1]
        
        # Default to single mode if no clusters are detected (e.g., small N or noise)
        if len(unique_labels) == 0:
            unique_labels = [0]
            labels = np.zeros(n_samples)
            
        n_modes = len(unique_labels)
        print(f"Detected {n_modes} mode(s) using {init_method} initialization.")
    
        # --- Step 4: Initial Mode Coordinate Extraction ---
        initial_means = []
        for label in unique_labels:
            mask = (labels == label)
            cluster_data_reduced = reduced_space[mask]
            centroid_reduced = np.mean(cluster_data_reduced, axis=0)
            
            # Method A: Mathematical Inverse Mapping (UMAP-based)
            if init_method == "inverse" and reducer is not None:
                try:
                    recon = reducer.inverse_transform(centroid_reduced.reshape(1, -1))
                    # Rescale from standardized units to physical units
                    initial_means.append(recon.flatten() * np.std(samples, axis=0) + np.mean(samples, axis=0))
                    continue
                except: 
                    pass # Fallback to index if inverse_transform fails
            
            # Method B: Nearest Sample Index Tracking (Medoid approach)
            dist = np.linalg.norm(cluster_data_reduced - centroid_reduced, axis=1)
            global_idx = np.where(mask)[0][np.argmin(dist)]
            initial_means.append(samples[global_idx])
        
        initial_means = np.array(initial_means)
    
        # --- Step 5: GMM Fitting in Original Parameter Space ---
        # Strong regularization (reg_covar) prevents LinAlgErrors when N < D
        reg_val = 1e-3 if n_samples < n_params else 1e-6
        gmm = GaussianMixture(
            n_components=n_modes, 
            means_init=initial_means,
            covariance_type="full",
            reg_covar=reg_val 
        )
        gmm.fit(samples)
    
        # --- Step 6: Point and Interval Estimation via Precision Analysis ---
        results = []
        for k in range(n_modes):
            mu_k = gmm.means_[k]
            sigma_k = gmm.covariances_[k]
            
            # Calculate Pseudo-inverse to handle rank-deficient matrices in small N cases
            # Precision Matrix K = Sigma^+
            precision_k = np.linalg.pinv(sigma_k)
            
            # Marginal Error (Shadow on axis): sqrt(diag(Sigma))
            marginal_std = np.sqrt(np.maximum(np.diag(sigma_k), 0))
            
            # Conditional Error (Chord through the mode): 1 / sqrt(diag(K))
            # Clipping diag(K) to prevent division by zero or negative variance artifacts
            k_diag = np.diag(precision_k)
            k_diag[k_diag <= 0] = np.inf 
            conditional_std = 1.0 / np.sqrt(k_diag)
            
            results.append({
                "weight": gmm.weights_[k],
                "point_estimate": mu_k,
                "marginal_error": marginal_std,
                "conditional_error": conditional_std
            })
    
        return results
    
    def save(self):
        return None
    