import bayesbay as bb
import numpy as np
import pandas as pd


class PostAnalyzer:
    """Analyze posterior sampling results.
    
    The sampling results is the form of BayesBay output.
    However, BayesBay is not needed to analyze.
    """
    
    def __init__(self, result_df: pd.DataFrame, concatenate_chains: bool = True):
        """
        Parameters
        ----------
        result_df : pd.DataFrame
            This should be the return of bayesbay.BayesianInversion.get_results
            or at least the same form of that.
        concatenate_chains : bool, optional
            Same argum
            The default is True.

        Returns
        -------
        None.
        """
        self.result_df = result_df
        
        df_with_id = df.copy()
        df_with_id.index.name = 'origin_grp_id'
        df_with_id = df_with_id.reset_index()
        
        # 2. 모든 열(기존 열 + id 열)에 대해 explode 실행
        # 'origin_grp_id'는 스칼라 값이므로 explode 시 자동으로 복제(Broadcasting)됩니다.
        target_cols = df.columns.tolist()
        df_exploded = df_with_id.explode(target_cols).reset_index(drop=True)
        
    
    def estimate():
        return None
    
    def save():
        return None

class Organizer:
    """Organize sampling results of Bayesbay."""
    
    def __init__(self, inversion: bb.BayesianInversion):
        results = inversion.get_results(concatenate_chains=False)
        
        self.sample_res = {}
        self.est_res = {}
        # Collect information of parameter spaces: Space names, parameter names, dimensions.        
        self.trans_space_info = {}
        self.fixed_space_info = {}
        for key, val in inversion.parameterization.parameter_spaces.items():
            # "key" is a space name.
            # Trans-spaces have None dimensions but have min and max.
            if val._n_dimensions is None:
                self.trans_space_info[key] = {
                    "params": list(val.parameters.keys()),
                    "n_dims_samples": self.results[f"{key}.n_dimensions"],
                    "unique_n_dims": np.unique(self.results[f"{key}.n_dimensions"]),
                    "n_dims_min": val.n_dimensions_min,
                    "n_dims_max": val.n_dimensions_max,
                }
            # Fixed-spaces have integer numbers of dimensions.
            elif isinstance(val._n_dimensions) is int:
                self.fixed_space_info[key] = {
                    "params": list(val.parameters.keys()),
                    "n_dims_fixed": val.n_dimensions,
                }
        
    def arrange(self, sort_ref):
        """Arrange posterior samples of parameters and forward model values.
            
        Sorting by order of trans-parameteres ('sort_ref') and the number of dimensions.
        """
        all_trans_params_dims = {key: val["unique_n_dims"] for key, val in self.trans_space_info.items()}
        for key, val in self.trans_space_info.items():
            for dim in val["unique_n_dims"]:
                dim = dim.item()  # Not nescessary but just for Spyder GUI
                self.sample_res[dim] = {}
                idxs = np.where(self.n_dims == dim)[0]
                # Collect posterior samples
                for param in val["params"]:
                    samples = []
                    for idx in idxs:
                        samples.append(self.results[f"{key}.{param}"][idx])
                    samples = np.vstack(samples)
                    # Save sorting indices refered by the parameter "sort_ref"
                    if param == sort_ref:
                        sort_idxs = np.argsort(samples, axis=1)
                    self.sample_res[dim][pp] = samples
                # Sort posterior samples
                for tn in trans_param_names:
                    samples = self.sample_res[dim][tn]
                    self.sample_res[dim][tn] = np.take_along_axis(samples, sort_idx, axis=1)
                
                fwd_vals = []
                for idx in idxs:
                    fwd_vals.append(results["target.dpred"][idx])
                fwd_vals = np.vstack(fwd_vals)
                
        for key, val in self.fixed_space_info.items():
            for param in val["params"]:
                samples = []
                for i in idxs:
                    samples.append(self.results[f"{key}.{param}"][i].item())
                self.sample_res[fn] = np.array(samples)                     
        
        return None
    """
    def estimate(self):
        # For trans parameters
        for dim in self.unique_dims:
            dim = dim.item()  # This is not nescessary but just to solve Spyder Variable Explorer GUI problem.
            self.sample_res[dim] = {}
            idxs = np.where(self.n_dims == dim)[0]
            # Collect posterior samples
            if "trans_space" in self.param_space_names.keys():
                for tn in self.param_space_names[]:
                    samples = []
                    for i in idxs:
                        samples.append(results[f"trans_space.{tn}"][i])
                    samples = np.vstack(samples)
                    # Save sorting index
                    if tn == sort_ref:
                        sort_idx = np.argsort(samples, axis=1)
                    est_results["samples"][dim][tn] = samples
                # Sort posterior samples
                for tn in trans_param_names:
                    samples = est_results["samples"][dim][tn]
                    est_results["samples"][dim][tn] = np.take_along_axis(samples, sort_idx, axis=1)
                # Estimate parameters
                est_results["estimates"][dim] = {}
                for tn in trans_param_names:
                    est_results["estimates"][dim][tn] = np.vstack((
                        np.mean(est_results["samples"][dim][tn], axis=0),
                        np.percentile(est_results["samples"][dim][tn], [50, 15.87, 84.13], axis=0),
                    ))
                
                # For forward model values
                fwd_vals = []
                for i in idxs:
                    fwd_vals.append(results["target.dpred"][i])
                fwd_vals = np.vstack(fwd_vals)
                est_results["samples"][dim]["forward"] = fwd_vals
                est_results["estimates"][dim]["forward"] = np.vstack((
                    np.mean(fwd_vals, axis=0),
                    np.percentile(fwd_vals, [50, 15.87, 84.13], axis=0),
                    ))

        # For fixed parameters
        for fn in fixed_param_names:
            samples = []
            for i in idxs:
                samples.append(results[f"fixed_space.{fn}"][i].item())
            est_results["samples"][fn] = np.array(samples)     
            est_results["estimates"][fn] = np.hstack((
                np.mean(est_results["samples"][fn], axis=0),
                np.percentile(est_results["samples"][fn], [50, 15.87, 84.13], axis=0),
                ))
        return None
    """
    def output(self):
        return None