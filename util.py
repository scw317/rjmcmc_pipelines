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
    
    def __init__(self, postsamples: pd.DataFrame, sort_refs: list[str], concatenate_chains: bool = True):
        """
        Parameters
        ----------
        postsamples : pd.DataFrame
            This should be the return of bayesbay.BayesianInversion.get_results
            or at least the same form of that.
        sort_refs : list[str]
            Sorting reference parameter names for each space.
            A regular expression pattern of parameter is '{space}.{param}'.
        concatenate_chains : bool, optional
            The same parameter as bayesbay.BayesianInversion.get_results. The default is True.
            Decide whether samples from all chains in 'postsamples' are aggregated or seperated.
            
        Returns
        -------
        None.
        """
        if concatenate_chains:
            if "chain" in postsamples.columns:
                self.postsamples = postsamples
            else:
                # Add the meaningless identical chain number (0) column just for pipeline
                chain_col = pd.Series(np.zeros(len(postsamples)), name="chain")
                self.postsamples = pd.concat((chain_col, postsamples), axis=1)
        else:
            # Numbering chains from indices of the original dataframe
            df_with_id = postsamples.copy()
            df_with_id.index.name = "chain"
            # The original indices becomes the new column named "chain".
            df_with_id = df_with_id.reset_index()
            # Explode chain-wise elements for all columns including the "chain" column
            self.postsamples = df_with_id.explode(postsamples.columns.tolist()).reset_index(drop=True)

        self.get_schema()
        #self.sort_df(sort_refs)
    
    def get_schema(self):
        # Add original column names for reference
        schema = pd.Series(self.postsamples.columns, name="col_name", dtype=str)

        # Regex logic: 
        # field: Space or target names (everything before the last dot)
        # attr: Parameter or attribute names (after the last dot)
        # This handles "{space}.{param}", "{space}.n_dimensions" and f"{target}.dpred".
        pattern = r"^(?P<field>.*)\.(?P<attr>.*)$"
        schema = pd.concat((schema, schema.str.extract(pattern)), axis=1)
        
        # Pre-calculate the set of trans-dimensional spaces
        # A space is trans-dimensional if it contains "n_dimensions" anchor
        trans_spaces = set(schema.loc[schema["attr"] == "n_dimensions", "field"])
        
        # Define categorization conditions and corresponding values
        # Priority is strictly enforced by the order of the list
        conditions = [
            schema["field"].isna() & schema["attr"].isna(),  # Rule 1: Fallback for no split
            schema["attr"] == "n_dimensions",  # Rule 2: Dimension count column
            (schema["field"].isin(trans_spaces)) & (schema["attr"] != "n_dimensions"),  # Rule 3: Trans params
            schema["attr"] == "dpred"  # Rule 4: Forward model output
        ]
        
        # Outpus following conditions
        outputs = [schema["col_name"], "dim", "trans", "target"]
        
        # Apply selection logic with "fixed" as the default case
        schema["cat"] = np.select(conditions, outputs, default="fixed")
        
        self.schema = schema

    def align_all_sample_sets(list_of_sample_arrays, params_per_flare=2):
        """
        Align multiple sample sets with different flare counts to a master reference.
        The master reference is chosen as the last sample of the set with maximum flares.
        
        Parameters:
        - list_of_sample_arrays: list of np.ndarray, each with shape (n_samples_i, n_params_i)
        - params_per_flare: int, number of parameters per single flare (e.g., 2 for [t, A])
        """
        
        # --- Step 1: Identify Master Reference ---
        # Find the maximum number of parameters across all sets
        max_params = max(arr.shape[1] for arr in list_of_sample_arrays)
        
        # Filter sets that have the maximum number of parameters
        candidate_sets = [arr for arr in list_of_sample_arrays if arr.shape[1] == max_params]
        
        # Pick the last sample of the last candidate set as the master reference
        # master_ref_flat: (max_params,)
        master_ref_flat = candidate_sets[-1][-1]
        n_flares_max = max_params // params_per_flare
        master_ref = master_ref_flat.reshape(n_flares_max, params_per_flare)
        
        print(f"Master reference selected with {n_flares_max} flares.")
    
        final_aligned_list = []
    
        # --- Step 2: Align each set to the Master Reference ---
        for arr_idx, samples in enumerate(list_of_sample_arrays):
            n_samples, n_params = samples.shape
            n_flares_curr = n_params // params_per_flare
            
            # Initialize aligned array for the current set with NaNs
            # Shape: (n_samples, n_flares_max * params_per_flare)
            aligned_set = np.full((n_samples, n_flares_max, params_per_flare), np.nan)
            
            for i in range(n_samples):
                curr_sample = samples[i].reshape(n_flares_curr, params_per_flare)
                
                # Step 3: Rectangular Hungarian Matching
                # Cost Matrix size: (n_flares_max x n_flares_curr)
                # This identifies which of the current flares best fits the master slots
                cost_matrix = cdist(master_ref, curr_sample, metric='euclidean')
                
                # Find optimal assignment
                # row_ind: Master Index (0 to K_max-1), col_ind: Current Sample Index
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                # Map parameters to the fixed master slots
                for r_idx, c_idx in zip(row_ind, col_ind):
                    aligned_set[i, r_idx] = curr_sample[c_idx]
            
            # Flatten back to (n_samples, max_params)
            final_aligned_list.append(aligned_set.reshape(n_samples, -1))
            print(f"Set {arr_idx} aligned. (Flares: {n_flares_curr} -> {n_flares_max})")
    
        return final_aligned_list, master_ref_flat
                                
    def sort_df(self, sort_refs):
        """Element-wise sorting by order of parameters in sort_refs."""
        
        def sort_df_by_dim(df: pd.DataFrame):
            ref_col= np.array(df[sort_refs].tolist())
            indices = np.argsort(ref_col, axis=1)
            sorted_df = pd.DataFrame(index=df.index)
            def wrapper(x):
                arr = np.array(x.tolist())
                return np.take_along_axis(arr, indices, axis=1)
            sorted_df[list(self.trans_cols)].apply(wrapper)
            
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
    