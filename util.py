import warnings
from pathlib import Path

import arviz as az
import bayesbay as bb
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from umap import UMAP


def get_unique_path(path: Path | str) -> Path:
    """Generate a unique file path by appending an incrementing number if the file exists.
    
    Format: {path}_{num}.{suffix}
    """
    path = Path(path)
    
    # If path does not exist, return it immediately
    if not path.exists():
        return path

    parent = path.parent
    stem = path.stem
    suffix = path.suffix
    
    counter = 1
    while True:
        # Construct new path with incremented number
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


def get_column_schema(postsamples: pd.DataFrame) -> pd.DataFrame:
    """Get column schema of the standard form of BayesBay posterior sample results.
    
    Recognize which column represents dimension features, targets,
    or parameters of trans or fixed dimensional space, and so on.
    """
    # Add original column names for reference
    schema = pd.Series(postsamples.columns, name="col_name", dtype=str)

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
    
    return schema


def sort_and_match_array(
        array: np.ndarray,
        ref_idx: int | None = None,
        match: bool = True,
        ) -> np.ndarray:
    """Sort by a specific parameter and match sample by sample.
    
    Use Hungarian matching based on standardized-Euclidean metric.

    Parameters
    ----------
    array : np.ndarray
        Shape: (samples, dimensions, params) == (N, K, P).
    ref_idx : int | None, optional
        A sorting reference index. The default is None.
    match : bool, optional
        Do match or not. The default is True.

    Returns
    -------
    aligned_array : np.ndarray
        Sorted and matched.
    """
    # Sort along a dimension axis (axis=1) by a specific parameter (ref_idx in axis=2)
    if ref_idx is not None:
        sort_idxs = np.argsort(array[..., ref_idx], axis=1)  # Shape: (N, K)
        array = np.take_along_axis(array, sort_idxs[..., np.newaxis], axis=1)
    
    if not match:
        return array

    # Calculate variance for "seuclidean" metric for each parameter
    variances = np.var(array, axis=(0, 1))  # Shape: (P,)
    variances[variances == 0] = 1.0  # Avoid division by zero
    
    aligned_array = np.zeros_like(array)
    aligned_array[-1] = np.mean(array, axis=0)  # Last is reference to calculate metric.
    
    # Perform Hungarian matching for each sample
    for i in range(array.shape[0] - 1):
        current_sample = array[i]  # Shape: (K, P)
        
        # Compute cost with standardized Euclidean distance refered by the last sample
        cost_matrix = cdist(aligned_array[-1], current_sample, metric="seuclidean", V=variances)
        
        # Solve optimal assignment
        _, col_idx = linear_sum_assignment(cost_matrix)

        # Record the aligned current sample
        aligned_array[i] = current_sample[col_idx]
    
    return aligned_array


def align_parameters(
        postsamples: pd.DataFrame,
        schema: pd.DataFrame,
        sort_refs: list[str] | None = None,
        ) -> pd.DataFrame:
    """Align parameters to solve the label-switching problem.
    
    Parameters
    ----------
    postsamples : pd.DataFrame
    schema : pd.DataFrame
    sort_refs : list[str] | None, optional
        Sorting reference parmaters for each trans-dimensional space.
        If it is None, not sorted. The default is None.
        ["{space}.{param}", ...].
    
    Returns
    -------
    df : pd.DataFrame
        Aligned.
    """
    df = postsamples

    # Categories definition ("trans" or "fixed", is "trans")
    categories = [("trans", True), ("fixed", False)]
    
    for cat, is_trans in categories:
        if is_trans:
            space_groups = schema.loc[schema["cat"]=="dim", ["field", "col_name"]].values
        else:
            fields = np.unique(schema.loc[schema["cat"]=="fixed", "field"])
            space_groups = [(f, None) for f in fields]  # No dim_col (None)

        for space, dim_col in space_groups:
            param_mask = (schema["field"] == space) & (schema["cat"] == cat)
            param_col_names = schema.loc[param_mask, "col_name"].tolist()
            
            if not param_col_names:
                continue

            # Extract ref_idx in param_col_names for sorting reference
            ref_idx = None
            if sort_refs:
                for i, p_name in enumerate(param_col_names):
                    if p_name in sort_refs:
                        ref_idx = i
                        break

            # Groups by dimensions for trans-dimensional and total dataframe for fixed-dimensional
            groups = df.groupby(dim_col) if is_trans else [("fixed", df)]
            
            for _, group in groups:
                if len(group) == 0:
                    continue
                
                # Shape: (samples, dimensions, params) == (N, K, P)
                data_array = np.stack(
                    [np.stack(group[param].to_numpy()) for param in param_col_names],
                    axis=-1,
                )
                
                aligned_array = sort_and_match_array(data_array, ref_idx)

                # Write back to the original dataframe in-place
                for p, param in enumerate(param_col_names):
                    df.loc[group.index, param] = pd.Series(
                        list(aligned_array[..., p]), index=group.index
                    )

    return df


def orginize_results(
        inversion: bb.BayesianInversion,
        save_dir: Path | str | None = None,
        sort_refs: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Organize sampling results and acceptance statistics.
    
    Parameters
    ----------
    inversion : bb.BayesianInversion
    save_dir : Path | str | None, optional
        Save directory path. If it is None, returns are not saved.
        The default is None.
    sort_refs: list[str] | None = None,
        Sorting reference parmaters for each trans-dimensional space.
        
    Retruns
    -------
    postsamples : pd.DataFrame
        Posterior samples.
    acceptances : pd.DataFrame
        The numbers of proposed and accepted samples and their ratio.
    """
    markov_chains = inversion.chains  #  bb.MarkovChain instance
    
    # Get acceptance statistics and posterior samples from markov_chains
    acceptance_list = []
    postsample_list = []
    for chain in markov_chains:
        # Acceptance statistics
        stats = chain.statistics
        
        proposed_dict = {"chain_id": chain.id, "kind": "proposed"}
        accepted_dict = {"chain_id": chain.id, "kind": "accepted"}
        ratio_dict = {"chain_id": chain.id, "kind": "ratio"}
        
        # The total numbers of proposed and accepted samples
        proposed_dict["total"] = stats["n_proposed_models_total"]
        accepted_dict["total"] = stats["n_accepted_models_total"]
        # Total acceptance ratio
        ratio_dict["total"] = stats["n_accepted_models_total"] / stats["n_proposed_models_total"]
        
        # The numbers of proposed and accepted samples for each space
        for dict_p, dict_a in zip(stats["n_proposed_models"].values(), stats["n_accepted_models"].values()):
            proposed_dict.update(dict_p)
            accepted_dict.update(dict_a)
            # Acceptance ratios for each space
            for (key, val_p), val_a in zip(dict_p.items(), dict_a.values()):
                ratio_dict[key] = val_a / val_p
        
        acceptance_df = pd.DataFrame((proposed_dict, accepted_dict, ratio_dict))
        acceptance_list.append(acceptance_df)
        
        # Posterior samples
        postsample_df = pd.DataFrame(inversion.get_results_from_chains(chain))
        
        # Add the acceptance ratio column
        acceptance_ratio_col = pd.Series(np.full(len(postsample_df), ratio_dict["total"]), name="acceptance_ratio")
        postsample_df = pd.concat((acceptance_ratio_col, postsample_df), axis=1)
        
        # Add the chain ID column
        chain_id_col = pd.Series(np.full(len(postsample_df), chain.id), name="chain_id")
        postsample_df = pd.concat((chain_id_col, postsample_df), axis=1)
        
        postsample_list.append(postsample_df)   
        
    acceptances = pd.concat(acceptance_list, ignore_index=True)
    postsamples = pd.concat(postsample_list, ignore_index=True)
    
    # Align parameters to solve the label-switching problem
    schema = get_column_schema(postsamples)
    postsamples = align_parameters(postsamples, schema, sort_refs)
    
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        postsample_save_path = get_unique_path(Path(save_dir) / "postamples.parquet")
        postsamples.to_parquet(path=str(postsample_save_path), engine="pyarrow", compression="zstd")
        acceptance_save_path = get_unique_path(Path(save_dir) / "acceptances.csv")
        acceptances.to_csv(str(acceptance_save_path))
    
    return postsamples, acceptances


class PostProcess:
    """Process and analyze posterior sampling results.
    
    The sampling results should the form of BayesBay output,
    however, the BayesBay library is not needed explicitly.
    """
    
    def __init__(self, postsamples: pd.DataFrame, concatenate_chains: bool = True):
        """
        Parameters
        ----------
        postsamples : pd.DataFrame
            This should be the return of bb.BayesianInversion.get_results
            or at least the same form of that.
            Highly recommended to use the return of organize_results.
        concatenate_chains : bool, optional
            The same parameter as bb.BayesianInversion.get_results. The default is True.
            Decide whether samples from all chains in 'postsamples' are aggregated or seperated.
            
        Returns
        -------
        None.
        """
        if concatenate_chains:
            if "chain_id" in postsamples.columns:
                self.postsamples = postsamples
            else:
                # Add the meaningless identical chain ID (0) column just for pipeline
                chain_id_col = pd.Series(np.zeros(len(postsamples)), name="chain_id")
                self.postsamples = pd.concat((chain_id_col, postsamples), axis=1)
        else:
            # Numbering chains from indices of the original dataframe
            df_with_id = postsamples.copy()
            df_with_id.index.name = "chain_id"
            # The original indices becomes the new column named "chain_id".
            df_with_id = df_with_id.reset_index()
            # Explode chain-wise elements for all columns including the "chain_id" column
            self.postsamples = df_with_id.explode(postsamples.columns.tolist()).reset_index(drop=True)

        self.get_schema()
    
    def get_schema(self):
        self.schema = get_column_schema(self.postsamples)

    def align(self):
        """Hungarian matching to solve the label-switching problem."""
        self.postsamples = align_parameters(self.postsamples, self.schema)
    
    def to_array(self, stack_chains, concat_params, to_3d) -> (dict[tuple[int, ...], np.ndarray], list[list[str]]):
        """Convert self.postsamples into np.ndarray for each combination of dimensions.
        
        Parameters
        ----------
        stack_chains : bool
            Stack chians as a new array axis.
            If each chain has the different number of draws, trimming draws.
            A draw is samples for each chain.
        concat_params : bool
            Concatenate parameters as a same array axis.
        to_3d : bool
            Arrays become 3d which is compatible to arviz.
    
        Returns
        -------
        sample_arrays_by_dims : dict[tuple[int, ...], np.ndarray]
            The keys are combinations of dimensions.
            The values are posterior samples arrays.
        sample_array_schema : list[list[str]]
            [["{space}.n_dimensions", ...], ["{space}.{param}", ...]].
        
        Notes
        -----
        For arviz, recommend stack_chains = True, concat_params = False, and to_3d = True. 
        For GMM, recommend stack_chains = False, concat_params = True, and to_3d = False.
        
        Examples
        --------
        if stack_chains == True and concat_params == True:
            sample_arrays_by_dims = {(dims,): array(chains, draws, all params)}.
        if stack_chains == True and concat_params == False:
            sample_arrays_by_dims = {(dims,): {param: array(chains, draws, param)}}.
        if stack_chains == False and concat_params == True:
            sample_arrays_by_dims = {(dims,): array(samples, all params)}.
        if stack_chains == False and concat_params == False:
            sample_arrays_by_dims = {(dims,): {param: array(samples, param)}.
        """
        df = self.postsamples
        schema = self.schema
        
        dim_col_names = schema.loc[schema["cat"]=="dim", "col_name"].tolist()  # "{space}.n_dimensions"
        param_col_names = schema.loc[schema["cat"].isin(["trans", "fixed"]), "col_name"].tolist()  # "{space}.{param}"
        
        # Group by dimensions if trans-dimensional spaces are exist.
        # If not, the original dataframe is the only group itself.
        groups = df.groupby(dim_col_names) if dim_col_names else [(None, df)]
        
        sample_array_schema = [dim_col_names, param_col_names]
        sample_arrays_by_dims = {}  # {(dims,): array}
        
        # If stacking chains have inhomogeneous draws, triming samples for each chain to be homogeneous.
        if stack_chains:
            for dims, group in groups:
                if len(group) == 0:
                    continue
                
                group = group.sort_values("chain_id")
                chain_ids = group["chain_id"].to_numpy()
                # Find all chain IDs and their sample (draw) numbers
                unique_chain_ids, chain_id_counts = np.unique(chain_ids, return_counts=True)
                # The minimum draw number of chains among all chain IDs
                min_draw = np.min(chain_id_counts)
                
                # Counting each chain ID in descending order to trim front draws
                # e.g., [3 2 1 0 2 1 0 4 3 2 1 0 ...] when chain_id_counts == [4 3 5 ...]
                descending_counter = np.concatenate([np.arange(count)[::-1] for count in chain_id_counts])
                # Masking False where descending counting exceeds the minimum
                chain_mask = descending_counter < min_draw
                
                if concat_params:
                    param_arrays = np.concatenate(
                        [np.stack(group[param].to_numpy()[chain_mask]) for param in param_col_names],  # Shape: [(samples, param)]
                        axis=1,
                    )  # Shape: (samples, all params)
                    param_arrays = param_arrays.reshape(len(unique_chain_ids), min_draw, -1)  # Shape: (chains, draws, all params)
                    
                else:
                    param_arrays = {
                        param: np.stack(group[param].to_numpy()[chain_mask]).reshape(len(unique_chain_ids), min_draw, -1)  # Shape: (chains, draws, param)
                        for param in param_col_names
                    }  # Shape: {param: (chains, draws, param)}
                sample_arrays_by_dims[dims] = param_arrays
        
        else:
            for dims, group in groups:
                if len(group) == 0:
                    continue
                
                if concat_params:
                    param_arrays = np.concatenate(
                        [np.stack(group[param].to_numpy()) for param in param_col_names],  # Shape: [(samples, param)]
                        axis=1,
                    )  # Shape: (sampls, all params)
                    
                else:
                    param_arrays = {param: np.stack(group[param].to_numpy()) for param in param_col_names}  # Shape: {param: (samples, param)}
                sample_arrays_by_dims[dims] = param_arrays
        
        # Make the shape {(dims,): array(1, samples, all params)}.
        if to_3d and (not stack_chains) and concat_params:
            for key, val in sample_arrays_by_dims.items():
                sample_arrays_by_dims[key] = val[np.newaxis, ...]
                
        # Make the shape {(dims,): {param: array(1, samples, param)}}
        if to_3d and (not stack_chains) and (not concat_params):
            for dim_key, in_dict in sample_arrays_by_dims.items():
                for key, val in in_dict.items():
                    sample_arrays_by_dims[dim_key][key] = val[np.newaxis, ...]
        
        return sample_arrays_by_dims, sample_array_schema
    
    def by_arviz(self, save_dir):
        samples_by_dims, sample_schema = self.to_array(
            stack_chains=True, concat_params=False, to_3d=True
        )

        results = {}
        for dims, samples in samples_by_dims.items():
            results[dims] = az.from_dict({"posterior": samples})
        
        if save_dir:
            arviz_save_dir = get_unique_path(Path(save_dir) / "arviz")
            arviz_save_dir.mkdir(parents=True, exist_ok=True)
            
            for dims, idata in results.items():
                # Save results by nc (hdf5)
                results_save_path = arviz_save_dir / f"results_dim{dims}.nc"
                idata.to_netcdf(str(results_save_path))
                
                # Save summaries by csv
                summary_save_path = arviz_save_dir / f"summary_dim{dims}.csv"
                summary_df = az.summary(idata, ci_kind="hdi", round_to="none")
                summary_df.to_csv(str(summary_save_path))
                
                # Save posterior and trace plots
                post_plot_save_dir = arviz_save_dir / f"post_plot_dim{dims}"
                post_plot_save_dir.mkdir(parents=True, exist_ok=True)
                trace_plot_save_dir = arviz_save_dir / f"trace_plot_dim{dims}"
                trace_plot_save_dir.mkdir(parents=True, exist_ok=True)
                for param in list(idata.posterior.data_vars):
                    post_plot_save_path = post_plot_save_dir / f"{param}.png"
                    post = az.plot_dist(idata, var_names=param)
                    post.savefig(str(post_plot_save_path))
                    trace_plot_save_path = trace_plot_save_dir / f"{param}.png"
                    trace = az.plot_trace(idata, var_names=param)
                    trace.savefig(str(trace_plot_save_path))
                
        return results
    
    def by_gmm_for_dim(self, samples, var_threshold: float = 0.95, init_method="index"):
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
    
    def by_gmm(self):
        return None
    
    def get_results(self, save_dir, by: str = "both"):
        
        save_dir = Path(save_dir)
        
        return None
    