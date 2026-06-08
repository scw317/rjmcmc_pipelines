import csv
import pickle
from datetime import datetime
from inspect import signature
from pathlib import Path
from typing import Any, Sequence

import arviz as az
import bayesbay as bb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_list_like
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import median_abs_deviation

# Signature keyword arguments of bb.BayesianInversion.run()
SIGNATURE_BB_RUN_KEYS = list(signature(bb.BayesianInversion.run).parameters.keys())


def expand_array_columns(df: pd.DataFrame, use_multiindex: bool) -> pd.DataFrame:
    """Detects list-like cell columns and expands them.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame which include list-like cell columns. 
    use_multiindex : bool
        If True, create multi-index column DataFrame.
        Columns are hierarchical tuples. e.g., ('col', 0), ('col', 1). 
        If False, create single-index column DataFrame.
        Columns are flat names. e.g., 'col[0]', 'col[1]'.
        
    Returns
    -------
    pd.DataFrame
        Column-wise expanded DataFrame by decomposing list-like cells.
    
    Examples
    --------
    >>> df
        col_0    col_1
    0  [1, 2]   [3, 4]
    >>> expand_array_columns(df, True)
        col_0    col_1
        0   1    0   1
    0   1   2    3   4
    >>> expand_array_columns(df, False)
        col_0[0]  col_0[1]  col_1[0]  col_1[1]
    0          1         2         3         4  
    """
    new_parts: list[pd.DataFrame] = []
    
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
                temp_col.columns = pd.MultiIndex.from_tuples([(col, "-")])
            new_parts.append(temp_col)
            
    return pd.concat(new_parts, axis=1)


def sort_and_match_array(array: np.ndarray, ref_idx: int | None = None) -> np.ndarray:
    """Sorting and Hungarian matching.

    Sorting along the axis=-2 refered by an index in axis=-1.
    Hungarian matching based on standardized Euclidean metric.

    Parameters
    ----------
    array : np.ndarray
        array.shape == (N, K, P).
        Sort and match indices in axis=1 among different indices in axis=0.
    ref_idx : int | None, optional
        The sorting reference index of axis=-1 for array. 
        If None, no sorting. The default is None.

    Returns
    -------
    aligned_array : np.ndarray
        Sorted and matched array.
    """
    # Reference sample for Hungarian matching
    reference = np.median(array, axis=0)
    
    # Sort the reference samples by ref_idx
    if ref_idx is not None:
        sort_idxs = np.argsort(reference[..., ref_idx])
        reference = reference[sort_idxs]
    
    # Calculate the effective variance for each index in axis=2. Shape: (P,).
    # This is robuster than standard deviation to outliner.
    variances = (1.4826 * np.median(median_abs_deviation(array, axis=0), axis=0))**2
    variances[variances == 0] = 1.0  # Avoid zero-division
    
    aligned_array = np.empty(array.shape)
    
    # Perform Hungarian matching for each indexed array in axis=0
    for i in range(len(array)):
        # Compute cost by standardized Euclidean metric between i-th sample and reference.
        # cost_matrix[j, k] is the metric between array[i][j], reference[k].
        cost_matrix = cdist(array[i], reference, metric="seuclidean", V=variances)
        
        # Find row_ind and col_ind minimizing cost_matrix[row_ind, col_ind].sum()
        row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=False)

        # Record the aligned i-th sample minimizing sum of metric
        aligned_array[i] = array[i, col_ind]
    
    return aligned_array


def get_column_schema(postsamples: pd.DataFrame) -> pd.DataFrame:
    """Get column schema of the standard form of BayesBay results.
    
    Recognize which column represents dimension features, targets,
    or parameters of trans or fixed dimensional space, and so on.
    
    Examples
    --------
    >>> schema
                   col_name    field          attr     cat
    0   space1.n_dimensions   space1  n_dimensions     dim     
    1         space0.param0   space0        param0   fixed  
    2         space0.param1   space0        param1   fixed
    3         space1.param0   space1        param0   trans
    4         space1.param1   space1        param1   trans
    5         target0.dpred  target0         dpred  target
    """
    # Add original column names for reference
    schema = pd.Series(postsamples.columns, name="col_name", dtype=str)

    # Regex logic: 
    # field: Space or target names (everything before the last dot)
    # attr: Parameter or attribute names (after the last dot)
    # It handles column names like f"{space}.{param}", f"{space}.n_dimensions" and f"{target}.dpred".
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


def align_parameters(
        postsamples: pd.DataFrame,
        schema: pd.DataFrame,
        sort_refs: Sequence[str] | None = None,
    ) -> pd.DataFrame:
    """Align parameters to solve the label-switching problem.
    
    Parameters
    ----------
    postsamples : pd.DataFrame
    schema : pd.DataFrame
    sort_refs : Sequence[str] | None, optional
        Sorting reference parmaters for each trans-dimensional space.
        If it is None, not sorted. The default is None.
        e.g., [f'{space}.{param}', ...].
    
    Returns
    -------
    df : pd.DataFrame
        Aligned postsamples.
    """
    df = postsamples

    for cat in ["trans", "fixed"]:
        
        if cat == "trans":
            space_groups = schema.loc[schema["cat"]=="dim", ["field", "col_name"]].to_numpy()
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

            # Group by dimensions for trans-dimensional or use total DataFrame for fixed-dimensional
            groups = df.groupby(dim_col) if cat == "trans" else [("fixed", df)]
            
            for _, group in groups:
                if len(group) == 0:
                    continue
                
                # data_array.shape == (draws, dimensions, parameters)
                data_array = np.stack(
                    [np.stack(group[param].to_numpy()) for param in param_col_names], axis=-1,
                )
                
                aligned_array = sort_and_match_array(data_array, ref_idx)

                # Write back to the original DataFrame in-place
                for p, param in enumerate(param_col_names):
                    aligned_series = pd.Series(list(aligned_array[..., p]), index=group.index)
                    df.loc[group.index, param] = aligned_series

    return df


def organize_results(
        inversion: bb.BayesianInversion,
        sort_refs: Sequence[str],
        save_dir: Path | str,
    ) -> pd.DataFrame:
    """Organize sampling results and acceptance statistics.
    
    Parameters
    ----------
    inversion : bb.BayesianInversion
    sort_refs : Sequence[str]
        Sorting reference parmaters for each trans-dimensional space.
    save_dir : Path | str
        Save directory path.
        
    Retruns
    -------
    postsamples : pd.DataFrame
        Posterior samples.
        Only unity temperature samples are included.
    """
    markov_chains: list[bb.MarkovChain] = inversion.chains
    
    # Get acceptance statistics and posterior samples from markov_chains
    acceptance_list = []
    postsample_list = []
    for chain in markov_chains:
        proposed_dict = {"chain_id": chain.id, "temperature": chain.temperature, "kind": "proposed"}
        accepted_dict = proposed_dict.copy()
        accepted_dict["kind"] = "accepted"
        ratio_dict = proposed_dict.copy()
        ratio_dict["kind"] = "ratio"
        
        # Acceptance statistics
        stats: dict = chain.statistics
        
        # The total numbers of proposed and accepted samples
        proposed_dict["total"] = stats["n_proposed_models_total"]
        accepted_dict["total"] = stats["n_accepted_models_total"]
        # Total acceptance ratio
        ratio_dict["total"] = stats["n_accepted_models_total"] / stats["n_proposed_models_total"]
        
        # The numbers of proposed and accepted samples for each parameter space
        for dict_p, dict_a in zip(stats["n_proposed_models"].values(), stats["n_accepted_models"].values()):
            proposed_dict.update(dict_p)
            accepted_dict.update(dict_a)
            # Acceptance ratios for each parameter space
            for (key, val_p), val_a in zip(dict_p.items(), dict_a.values()):
                ratio_dict[key] = val_a / val_p
        
        acceptance_df = pd.DataFrame((proposed_dict, accepted_dict, ratio_dict))
        acceptance_list.append(acceptance_df)
        
        # Posterior samples for a Markovs chain
        postsample_dict: dict[str, list[int | np.ndarray]] = inversion.get_results_from_chains(chain)
        
        # It is possible that some chains do not have any unity temperature samples,
        # therefore, postsample_dict can be empty dict.
        if postsample_dict:
            postsample_df = pd.DataFrame(postsample_dict)
            
            # Add the chain ID column
            chain_id_col = pd.Series(np.full(len(postsample_df), chain.id), name="chain_id")
            
            postsample_df = pd.concat((chain_id_col, postsample_df), axis=1)
            postsample_list.append(postsample_df)  
    
    # Concatenate multiple chains results
    acceptances = pd.concat(acceptance_list, ignore_index=True)
    postsamples = pd.concat(postsample_list, ignore_index=True)
    
    # Align parameters to solve the label-switching problem
    schema = get_column_schema(postsamples)
    postsamples = align_parameters(postsamples, schema, sort_refs)
    
    if save_dir:
        acceptances.to_csv(Path(save_dir) / "acceptances.csv", index=False)
    
    return postsamples


class PostProcess:
    """Process and analyze posterior sampling results."""
    
    def __init__(self, postsamples: pd.DataFrame, save_dir: Path | str = "./results") -> None:
        """
        Parameters
        ----------
        postsamples : pd.DataFrame
            Postsamples returned by organize_results.
        save_dir : Path | str, optional
            Save directory path. The default is './results'.
            
        Returns
        -------
        None
        """
        self.postsamples = postsamples
        self.save_dir = Path(save_dir)
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)
        self.schema = get_column_schema(self.postsamples)
    
    def est_dims(self, do_plot: bool) -> pd.DataFrame:
        """Estimate the number of dimensions of parmeters.
        
        Basically, estimate the number of multi-component model.
        If several kinds of multi-component models exist,
        estimate each number of multi-component model for each kind.
        """
        # Dimension column names like f"{space}.n_dimensions"
        dim_col_names = self.schema.loc[self.schema["cat"]=="dim", "col_name"].tolist()
        dim_cols = self.postsamples[dim_col_names].to_numpy()  # Shape: (samples, spaces)
        
        # Dimensions bins edges for each space
        bins_edges = [np.arange(np.min(col), np.max(col) + 2) - 0.5 for col in dim_cols.T]
        # Multi-dimensional histogram to get joint mode of dimensions
        hists, edges = np.histogramdd(dim_cols, bins=bins_edges)
        # The index in hists which is the mode
        joint_argmax = np.unravel_index(np.argmax(hists), hists.shape)
        
        # All space dimensions DataFrame including joint and marginal modes 
        dim_mode_dict = {"kind": ["joint", "marginal"]}
        for d, arg in enumerate(joint_argmax):
            dim_mode_dict[dim_col_names[d]] = [(edges[d][arg] + edges[d][arg + 1]) / 2]
        
        # Dimensions for each space
        for d, dim in enumerate(dim_cols.T):
            uniques, counts = np.unique(dim, return_counts=True)
            
            # The highest posterior probability dimensions
            dim_mode = uniques[np.argmax(counts)]
            
            # Dimensions posterior distribution plot
            if do_plot:
                fig, ax = plt.subplots()
                ax.hist(dim, bins=bins_edges[d], color="C0", histtype="step")
                ax.axvline(dim_mode, color="red")
                ax.annotate(dim_mode, (dim_mode, 0))
                ax.set_xlabel(dim_col_names[d])
                ax.set_ylabel("The number of posterior samples")
                fig.savefig(self.save_dir / f"{dim_col_names[d]}.png", bbox_inches="tight")
                plt.close()
            
            # Dimension distribution DataFrame
            dim_df = pd.DataFrame({"dim": uniques, "count": counts})
            dim_df.to_csv(self.save_dir / f"{dim_col_names[d]}.csv", index=False)
            
            # Update the marginal mode to the all space dimensions DataFrame
            dim_mode_dict[dim_col_names[d]].append(uniques[np.argmax(counts)])
        
        dim_mode_df = pd.DataFrame(dim_mode_dict)
        dim_mode_df.to_csv(self.save_dir / "dim_mode.csv", index=False)
        
        return dim_mode_df
    
    def to_arviz(self, concat_chains: bool) -> dict[tuple[int, ...] | None, dict[str, np.ndarray]]:
        """Convert self.postsamples into compatible form to arviz.from_dict.
        
        Parameters
        ----------
        concat_chains : bool
            Determine to concatenate chains as a single chain.
            If True, statistics of chain convergency (e.g., R-hat), are not given.
        
        Returns
        -------
        arviz_dict_by_dims : dict[tuple[int, ...] | None, dict[str, np.ndarray]]
            Dictionary compatible to input of arviz.from_dict().
            None replaces tuple[int, ...] when all spaces are fixed-dimensional.
        
        Examples
        --------
        >>> arviz_dict_by_dims
        {
            (2, 5): {
                param0: [[[101, 201, 893, ...],
                param1: [[[0.2, 3.4, 1.5, ...],
                ...
            },
            ...
        }
        """
        df = self.postsamples
        schema = self.schema
        
        # Dimension column names like f"{space}.n_dimensions"
        dim_col_names = schema.loc[schema["cat"]=="dim", "col_name"].tolist()  
        # Parameter column names liek f"{space}.{param}"
        param_col_names = schema.loc[schema["cat"].isin(["trans", "fixed"]), "col_name"].tolist()  
        
        # Group by dimensions if trans-dimensional spaces are exist.
        # If not, the original DataFrame is the only group itself.
        groups = df.groupby(dim_col_names) if dim_col_names else [(None, df)]
        
        arviz_dict_by_dims = {}
        
        # Treat all samples as one long chain (1, samples, dimensions)
        if concat_chains:
            for dims, group in groups:
                if len(group) == 0:
                    continue
                param_arrays = {
                    param: np.stack(group[param].to_numpy())[None, ...] 
                    for param in param_col_names
                }
                arviz_dict_by_dims[dims] = param_arrays
                
        # If chains have inhomogeneous draws, triming draws for each chain to be homogeneous.
        else:
            for dims, group in groups:
                if len(group) == 0:
                    continue
                
                # Count draws per chains
                counts = group["chain_id"].value_counts()
                unique_ids = counts.index.to_numpy()
                draw_counts = counts.to_numpy()
                
                # valid_chain_mask[i, j]: bool = draw_counts[j] >= draw_counts[i]
                valid_chain_mask = draw_counts >= draw_counts[..., None]
                # The number of chains of which draws are more than draws_counts
                valid_chain_counts = np.count_nonzero(valid_chain_mask, axis=1)
                
                # Sample volumn is the product between the refernce number of draws
                # and the number of chains of which draws are more than the reference number.
                candidate_volumes = valid_chain_counts * draw_counts
                
                # The optimal index maximizing sample volume.
                opt_idx = np.argmax(candidate_volumes)
                 
                opt_n_draws = draw_counts[opt_idx]
                opt_chain_ids = unique_ids[valid_chain_mask[opt_idx]]
                
                # Filter to include only opt_chain_ids
                filtered_group = group[group["chain_id"].isin(opt_chain_ids)]
                
                # Build homogenously trimmed draws for each parameters
                param_arrays = {}
                for param in param_col_names:
                    # Stack draws array of shape (opt_n_draws, dimensions) for each chain
                    # into the array of shape (len(opt_chain_ids), opt_n_draws, dimensions).
                    param_arrays[param] = np.stack(
                        [
                            np.stack(draws[param].tolist())[-opt_n_draws:]  # Remain last draws
                            for _, draws in filtered_group.groupby("chain_id")
                        ]
                    )
                arviz_dict_by_dims[dims] = param_arrays
        
        return arviz_dict_by_dims
    
    def by_arviz(self, concat_chains: bool, do_plot: bool) -> None:
        """Get estimates and statistics by ArviZ library."""
        arviz_dict_by_dims = self.to_arviz(concat_chains)
        
        results = {}
        for dims, samples in arviz_dict_by_dims.items():
            # Transform results to xarray.DataTree
            results[dims] = az.from_dict({"posterior": samples})
        
        # Process posterior samples of homogeneous dimensions
        for dims, idata in results.items():
            # Save directory by dimensions
            dims_dir = self.save_dir / f"{dims}"
            dims_dir.mkdir(parents=True, exist_ok=True)
            
            # Mean and median estimates and statistics summary DataFrames
            mean_summary_df = az.summary(idata, kind="all", ci_kind="hdi", ci_prob=0.6827, round_to="none")
            median_summary_df = az.summary(idata, kind="all_median", ci_prob=0.6827, round_to="none")
            
            # Mode estimates
            mode_dataset = az.mode(idata, round_to="none").to_dataset()
            mode_series = pd.Series(
                np.concatenate([mode_dataset[var].values.flatten() for var in mode_dataset.data_vars]),
                name="mode", index=mean_summary_df.index
            )
            
            # Save concatenated DataFrame with estimates and statistics
            summary_df = pd.concat((mean_summary_df, median_summary_df, mode_series), axis=1)
            summary_df.to_csv(dims_dir / "summary.csv")
            
            # Save distribution, trace, auto-correlation, and convergency plots
            if do_plot:
                param_names = list(idata.posterior.data_vars)
                for name in param_names:
                    az.plot_dist(idata, var_names=name).savefig(dims_dir / f"dist_{name}.png", bbox_inches="tight")
                    az.plot_trace(idata, var_names=name).savefig(dims_dir / f"trace_{name}.png", bbox_inches="tight")
                    az.plot_autocorr(idata, var_names=name).savefig(dims_dir / f"autocorr_{name}.png", bbox_inches="tight")
                az.plot_convergence_dist(idata, var_names=param_names).savefig(dims_dir / "convergency.png", bbox_inches="tight")
                plt.close()


class InversionHandler:
    """Handle Bayesian inversion using Bayesbay."""
    
    def __init__(
            self,
            inversion: bb.BayesianInversion,
            sort_refs: Sequence[str] | None = None,
            save_dir: Path | str = "./results",
        ) -> None:
        """
        Parameters
        ----------
        inversion : bb.BayesianInversion
            bayesbay.BaseBayesianInversion instance.
        sort_refs : Sequence[str] | None
            Parameter name to be reference of sort.
        save_dir : Path | str, optional
            Results save directory. The default is './results'.
        
        Returns
        -------
        None
        """
        self.inversion = inversion
        self.sort_refs = sort_refs
        self.save_dir = Path(save_dir)
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)
        
        # Keyword arguments for inversion.run()
        self.run_kwargs: dict[str, Any] = {}
        
        # States as checkpoint to run
        self.states: list[list[bb.State]] | None = None
        
        # Posterior samples to be processed
        self.postsamples: pd.DataFrame | None = None
    
    def check_run_kwargs(self) -> None:
        """Check whether unexpected keyword arguments exist in self.run_kwargs."""
        # Find unexpected keys in self.run_kwargs
        unexpected_keys = []
        for key in self.run_kwargs.keys():
            if key not in SIGNATURE_BB_RUN_KEYS:
                unexpected_keys.append(key)
                
        # Raise and report a error        
        if unexpected_keys:
            raise TypeError(
                "Unexpected keyword arguments for bayesbay.BayesianInversion.run():"
                f"\n{unexpected_keys}"
            )

    def save_states(self, save_path: Path | str) -> None:
        """Save parameter states as an attribute and Pickle."""
        # Extract current states from Markov chains
        self.states = [chain.current_state for chain in self.inversion.chains]
        
        # Write the current states as Pickle
        with open(self.sub_dir / "states.pkl", "wb") as f:
            pickle.dump(self.states, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_states(self, load_path: Path | str) -> None:
        """Load parameter states from Pickle."""
        with open(load_path, "rb") as f:
            self.states = pickle.load(f)
    
    def run(self, **kwargs) -> None:
        """Run new inversion or continue from self.states."""
        # Check whetehr self.run_kwargs is valid
        self.check_run_kwargs()
        # Update inversion.run() keyward arguments
        self.run_kwargs.update(kwargs)
        
        # Re-define self.inversion with self.states
        if self.states is not None:
            self.inversion = bb.BayesianInversion(
                parameterization=self.inversion.parameterization,
                log_likelihood=self.inversion.log_likelihood,
                n_chains=len(self.states),
                walkers_starting_states=self.states,
            )

        self.inversion.run(**self.run_kwargs)
        
        # Sub-directory named by the current timestamp
        self.sub_dir = self.save_dir / Path(datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.sub_dir.mkdir(parents=True, exist_ok=True)
        
        # Save inversion.run() keyward arguments as CSV
        with open(self.sub_dir / "run_kwargs.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["key", "value"])
            for key, val in self.run_kwargs.items():
                writer.writerow([key, val])
    
        # Save the current states of Markov chains.
        self.save_states(self.sub_dir / "states.pkl")
    
    def process(
            self,
            concat_samples: bool,
            concat_chains: bool,
            do_arviz_plot: bool = False,
            use_multiindex: bool | None = None,
        ) -> None:
        """Process posterior samples and get results.
        
        Parameters
        ----------
        concat_samples : bool
            Concatenate posterior samples with the last.
        concat_chains : bool
            Concatenate Markov chains.
        do_arviz_plot : bool, optional
            Plot ArviZ plots and save them. The default is False.
            Note: Plots spend more time and memory than processing.
        use_multiindex : bool | None, optional
            When saving posterior samples as CSV, use multi-index.
            See more details in expand_array_columns().
            If None, not saved as CSV. The default is None.
            
        Returns
        -------
        None
        """
        # Get posterior samples by the latest inversion
        postsamples = organize_results(self.inversion, self.sort_refs, self.sub_dir)
        
        # Concatenate the current posterior samples with the latter one
        if concat_samples and self.postsamples is not None:
            self.postsamples = pd.concat((self.postsamples, postsamples), ignore_index=True)
        # Replace the last posterior samples to the latest
        else:
            self.postsamples = postsamples
        
        # Process posterior samples
        post_process = PostProcess(self.postsamples, self.sub_dir)
        post_process.est_dims(do_plot=True)
        post_process.by_arviz(concat_chains, do_arviz_plot)
        
        # Save posterior samples as Parquet
        self.postsamples.to_parquet(
            self.sub_dir / "postamples.parquet", engine="pyarrow", compression="zstd"
        )
        # Save as column-wise expanded CSV by decomposing multi-dimension parameter columns
        if use_multiindex is not None:
            expanded_postsamples = expand_array_columns(self.postsamples, use_multiindex)
            expanded_postsamples.to_csv(self.sub_dir / "postamples.csv", index=False)