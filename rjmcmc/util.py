from pathlib import Path
from typing import Sequence

import bayesbay as bb

import numpy as np
import pandas as pd
from pandas.api.types import is_list_like
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import median_abs_deviation


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
        array.shape == (N, K, P). N: samples, K: dimensions, P: parameters.
        Sort and match indices in axis=1 among different indices in axis=0.
    ref_idx : int | None, optional
        The sorting reference index of axis=-1 for array. 
        If None, no sorting. The default is None.

    Returns
    -------
    aligned_array : np.ndarray
        Sorted and matched array.
    """
    array = array.copy()

    # Pre-sorting each sample by ref_idx before computing reference sample
    # This prevents reference collapse under symmetric label switching.
    if ref_idx is not None:
        sort_idxs = np.argsort(array[..., ref_idx], axis=1)
        array = np.take_along_axis(array, sort_idxs[..., None], axis=1)

    # Reference sample for Hungarian matching
    sample_ref = np.median(array, axis=0)
    
    # Calculate the effective variance for each index in axis=2. Shape: (P,).
    # This is robuster than standard deviation to outliner.
    variances = (1.4826 * np.median(median_abs_deviation(array, axis=0), axis=0))**2
    variances[variances == 0] = 1.0  # Avoid zero-division
    
    aligned_array = np.empty(array.shape)
    
    # Perform Hungarian matching for each indexed array in axis=0
    for i in range(len(array)):
        # Compute cost by standardized Euclidean metric between i-th sample and reference.
        # cost_matrix[j, k] is the metric between array[i][j], reference[k].
        cost_matrix = cdist(array[i], sample_ref, metric="seuclidean", V=variances)
        
        # Find row_ind and col_ind minimizing cost_matrix[row_ind, col_ind].sum()
        row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=False)

        # Record the aligned i-th sample minimizing sum of metric
        aligned_array[i, col_ind] = array[i]
    
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
    df = postsamples.copy()

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