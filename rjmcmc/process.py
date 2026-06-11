from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util import get_column_schema


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