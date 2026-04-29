from pathlib import Path

import arviz as az
import bayesbay as bb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_list_like
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import median_abs_deviation


def get_unique_path(path: Path | str, abort_count: int = 1000) -> Path:
    """Get a unique path by appending an incrementing number if the file exists.
    
    Format: {path}_{num}.{suffix} but {suffix} is not necessary.
    
    Parameters
    ----------
    path : Path | str
    abort_count : int, optional
        Increment abort as reaching this. The default is 1000.
    
    Retruns
    -------
    new_path : Path
        New path with incremented or random number.
    """
    path = Path(path)
    
    # If path does not exist, return it immediately
    if not path.exists():
        return path

    parent = path.parent
    stem = path.stem
    suffix = path.suffix
    
    for counter in range(1, abort_count + 1):
        # Construct new path with incremented number
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
    
    # Construct new path with random number when counter reach abort_count
    counter = np.random.randint(abort_count + 1, 100 * abort_count)
    new_path = parent / f"{stem}_{counter}{suffix}"
    return new_path


def expand_array_columns(df: pd.DataFrame, use_multiindex: bool = False) -> pd.DataFrame:
    """Detects list-like columns and expands them.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    use_multiindex :
        If True, creates hierarchical columns. 
        If False, creates flat names (e.g., 'col[0]').
        The default is False.
        
    Returns
    -------
    pd.DataFrame
        Column-wise expanded dataframe by decomposing list-like columns.
    """
    new_parts = []
    
    for col in df.columns:
        # Detect if the column contains list-like elements (excluding strings)
        first_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
        
        if is_list_like(first_val) and not isinstance(first_val, (str, bytes)):
            # Expand array to multiple columns
            expanded = pd.DataFrame(df[col].tolist(), index=df.index)
            
            if use_multiindex:
                # The original column index become level 0 (macroscopic) index.
                expanded.columns = pd.MultiIndex.from_product([[col], range(expanded.shape[1])])
            else:
                expanded.columns = [f"{col}[{i}]" for i in range(expanded.shape[1])]
            
            new_parts.append(expanded)
        else:
            # Handle scalar columns
            temp_col = df[[col]]
            if use_multiindex:
                # Align scalar columns to 2-level index system for concatenation
                temp_col.columns = pd.MultiIndex.from_tuples([(col, "scalar")])
            new_parts.append(temp_col)
            
    return pd.concat(new_parts, axis=1)


def sort_and_match_array(
        array: np.ndarray,
        ref_idx: int | None = None,
        do_match: bool = True,
        ) -> np.ndarray:
    """Sorting and Hungarian matching.

    The sorting is performed along the axis=1 refered by one index in axis=2.
    The Hungarian matching is based on standardized Euclidean metric.

    Parameters
    ----------
    array : np.ndarray
        Shape: (N, K, P).
    ref_idx : int | None, optional
        The sorting reference index of axis=2. The default is None.
    do_match : bool, optional
        Determine to do matching or not. The default is True.

    Returns
    -------
    aligned_array : np.ndarray
        Sorted and matched.
    """
    # Sort along axis=1 by ref_idx in axis=2
    if ref_idx is not None:
        sort_idxs = np.argsort(array[..., ref_idx], axis=1)  # Shape: (N, K)
        array = np.take_along_axis(array, sort_idxs[..., np.newaxis], axis=1)
    
    if not do_match:
        return array

    # Calculate the effective variance for each index in axis=2. Shape: (P,)
    variances = (1.4826 * np.median(median_abs_deviation(array, axis=0), axis=0))**2
    #variances = np.median(np.var(array, axis=0), axis=0)  # Not bad 
    #variances = np.var(array, axis=(0, 1))  # Very unstable
    variances[variances == 0] = 1.0  # Avoid division by zero
    
    # Reference for metric caluclation
    reference = array[-1]
    
    aligned_array = np.zeros_like(array)
    
    # Perform Hungarian matching for each indexed array in axis=0
    for i in range(array.shape[0]):
        # Compute cost with standardized Euclidean metric between i-th array and reference
        cost_matrix = cdist(array[i], reference, metric="seuclidean", V=variances)
        
        # Solve optimal assignment
        row_dix, col_idx = linear_sum_assignment(cost_matrix)

        # Record the aligned i-th array
        aligned_array[i] = array[i][col_idx]
    
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
    df = postsamples  # Shallow copy

    for cat in ["trans", "fixed"]:
        if cat == "trans":
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

            # Group by dimensions for trans-dimensional or use total dataframe for fixed-dimensional
            groups = df.groupby(dim_col) if cat == "trans" else [("fixed", df)]
            
            for _, group in groups:
                if len(group) == 0:
                    continue
                
                # Shape: (samples, dimensions, parameters) == (N, K, P)
                data_array = np.stack(
                    [np.stack(group[param].to_numpy()) for param in param_col_names],
                    axis=-1,
                )
                
                aligned_array = sort_and_match_array(data_array, ref_idx)

                # Write back to the original dataframe in-place
                for p, param in enumerate(param_col_names):
                    aligned_series = pd.Series(list(aligned_array[..., p]), index=group.index)
                    df.loc[group.index, param] = aligned_series

    return df


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


def organize_results(
        inversion: bb.BayesianInversion,
        sort_refs: list[str] | None = None,
        save_dir: Path | str = "./results",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Organize sampling results and acceptance statistics.
    
    Parameters
    ----------
    inversion : bb.BayesianInversion
    sort_refs: list[str] | None = None,
        Sorting reference parmaters for each trans-dimensional space.
    save_dir : Path | str, optional
        Save directory path. The default is './results'.
        
    Retruns
    -------
    postsamples : pd.DataFrame
        Posterior samples. Note that only T=1 chains have them.
    expanded_postsamples : pd.DataFrame
        Column-wise expanded postsamples.
    acceptances : pd.DataFrame
        The numbers of proposed and accepted samples and their ratio.
    temperatures : pd.DataFrame
        Temperatures (T) of chains
    """
    markov_chains = inversion.chains  #  bb.MarkovChain instance
    
    # Get temperatures, acceptance statistics and posterior samples from markov_chains
    temperature_dict = {"chain_id": [], "temperature": []}
    acceptance_list = []
    postsample_list = []
    for chain in markov_chains:
        temperature_dict["chain_id"].append(chain.id)
        temperature_dict["temperature"].append(chain.temperature)
        
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
        
        # Posterior samples.
        # Note that non-unity temperature chains do not have posterior samples.
        if inversion.get_results_from_chains(chain):
            postsample_df = pd.DataFrame(inversion.get_results_from_chains(chain))
            
            # DEPRECATED: Add the acceptance ratio column
            #acceptance_ratio_col = pd.Series(np.full(len(postsample_df), ratio_dict["total"]), name="acceptance_ratio")
            #postsample_df = pd.concat((acceptance_ratio_col, postsample_df), axis=1)
            
            # Add the chain ID column
            chain_id_col = pd.Series(np.full(len(postsample_df), chain.id), name="chain_id")
            postsample_df = pd.concat((chain_id_col, postsample_df), axis=1)
            
            postsample_list.append(postsample_df)  
        
    temperatures = pd.DataFrame(temperature_dict)
    acceptances = pd.concat(acceptance_list, ignore_index=True)
    postsamples = pd.concat(postsample_list, ignore_index=True)
    
    # Align parameters to solve the label-switching problem
    schema = get_column_schema(postsamples)
    postsamples = align_parameters(postsamples, schema, sort_refs)
    
    # Column-wise expansion by decomposing multi-dimension parameter columns
    expanded_postsamples = expand_array_columns(postsamples)
    
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        postsamples.to_parquet(str(get_unique_path(Path(save_dir) / "postamples.parquet")), engine="pyarrow", compression="zstd")
        expanded_postsamples.to_csv(str(get_unique_path(Path(save_dir) / "expanded_postamples.parquet")))
        acceptances.to_csv(str(get_unique_path(Path(save_dir) / "acceptances.csv")))
        temperatures.to_csv(str(get_unique_path(Path(save_dir) / "temperatures.csv")))
    
    return postsamples, expanded_postsamples, acceptances, temperatures


class PostProcess:
    """Process and analyze posterior sampling results.
    
    The sampling results should the form of BayesBay output,
    however, the BayesBay library is not needed explicitly.
    """
    
    def __init__(self, postsamples: pd.DataFrame, concatenate_chains: bool = True, save_dir: Path | str = "./results"):
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
        save_dir : Path | str, optional
            Save directory path. The default is './results'.
            
        Returns
        -------
        None
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
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
    
    def est_dims(self) -> pd.DataFrame:
        dim_col_names = self.schema.loc[self.schema["cat"]=="dim", "col_name"].tolist()
        dim_cols = self.postsamples[dim_col_names].to_numpy()  # Shape: (samples, spaces)
        
        # Dimensions bins edges for each space
        bins_edges = [
            np.arange(np.min(col), np.max(col) + 2) - 0.5
            for col in dim_cols.T
        ]
        # Multi-dimensional histogram to get joint mode of dimensions
        hists, edges = np.histogramdd(dim_cols, bins=bins_edges)
        # The index in hists which is the mode
        joint_argmax = np.unravel_index(np.argmax(hists), hists.shape)
        
        # All space dimensions dataframe including joint and marginal modes 
        dim_mode_dict = {"kind": ["joint", "marginal"]}
        for d, arg in enumerate(joint_argmax):
            dim_mode_dict[dim_col_names[d]] = [(edges[d][arg] + edges[d][arg + 1]) / 2]
        
        # Dimensions for each space
        for d, dim in enumerate(dim_cols.T):
            uniques, counts = np.unique(dim, return_counts=True)
            
            # Dimensions histogram
            fig, ax = plt.subplots()
            ax.hist(dim, bins=bins_edges[d])
            ax.set_xlabel(dim_col_names[d])
            plt.show()
            fig.savefig(str(get_unique_path(self.save_dir / f"{dim_col_names[d]}.png")))
            plt.close(fig)
            
            # Dimension distribution dataframe
            dim_df = pd.DataFrame({"dim": uniques, "count": counts})
            dim_df.to_csv(str(get_unique_path(self.save_dir / f"{dim_col_names[d]}.csv")))
            
            # Update the marginal mode to the all space dimensions dataframe
            dim_mode_dict[dim_col_names[d]].append(uniques[np.argmax(counts)])
        
        dim_mode_df = pd.DataFrame(dim_mode_dict)
        dim_mode_df.to_csv(str(get_unique_path(self.save_dir / "dim_mode.csv")))
        
        return dim_mode_df
                        
    def by_arviz(self, stack_chains: bool = True) -> dict:
        """Get estimates and statistics by arviz library."""
        samples_by_dims, sample_schema = self.to_array(
            stack_chains=stack_chains, concat_params=False, to_3d=True
        )
        
        results = {}
        for dims, samples in samples_by_dims.items():
            results[dims] = az.from_dict({"posterior": samples})
        
        arviz_save_dir = get_unique_path(self.save_dir / "arviz")
        arviz_save_dir.mkdir(parents=True, exist_ok=True)
        
        for dims, idata in results.items():           
            # Mean and median estimates and statistics summary dataframes
            mean_summary_df = az.summary(idata, kind="all", ci_kind="hdi", ci_prob=0.6827, round_to="none")
            median_summary_df = az.summary(idata, kind="all_median", ci_prob=0.6827, round_to="none")
            
            # Mode estimates
            mode_dataset = az.mode(idata, round_to="none").to_dataset()
            mode_series = pd.Series(
                np.concatenate([mode_dataset[var].values.flatten() for var in mode_dataset.data_vars]),
                name="mode", index=mean_summary_df.index
            )
            
            # Save concatenated dataframe with estimates and statistics
            summary_df = pd.concat((mean_summary_df, median_summary_df, mode_series), axis=1)
            summary_df.to_csv(str(arviz_save_dir / f"summary_dim{dims}.csv"))
            
            # Save posterior, trace, and autocorrelation plots
            for param in list(idata.posterior.data_vars):
                az.plot_dist(idata, var_names=param).savefig(str(arviz_save_dir / f"post_dim{dims}_{param}.png"))
                az.plot_trace(idata, var_names=param).savefig(str(arviz_save_dir / f"trace_dim{dims}_{param}.png"))
                az.plot_autocorr(idata, var_names=param).savefig(str(arviz_save_dir / f"autocorr_dim{dims}_{param}.png"))
                
        return results
    
    def by_gmm(self):
        return None


class InversionRepeater:
    
    def __init__(
            self,
            inversion: bb.BayesianInversion,
            sort_refs: list[str] | None = None,
            save_dir: Path | str = "./results",
        ):

        self.inversion = inversion
        
        def run(self, **kwargs):
            self.inversion.run(**kwargs)
            
            postsamples, expanded_postsamples, acceptances, temperatures = organize_results(
                self.inversion, self.sort_refs, self.save_dir
            )

            post_process = PostProcess(postsamples, True, self.save_dir)
            post_process.est_dims()
            post_process.by_arviz()
        
        return None
    
    