import numpy as np
import pandas as pd


class PostProcess:
    """Organize and analyze posterior sampling results.
    
    The sampling results should the form of BayesBay output,
    however, the BayesBay library is not needed explicitly.
    """
    
    def __init__(self, result_df: pd.DataFrame, concatenate_chains: bool = True):
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
        
        # Among the dataframe columns, find names of parameters and their spaces and targets
        cols = result_df.columns
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
        
    def estimate(self):
        return None
    
    def save(self):
        return None
    