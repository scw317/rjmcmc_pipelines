import numpy as np
import pandas as pd


class Organizer:
    
    def __init__(self, results: dict):
        self.results = results
        self.sample_res = {}
        self.est_res = {}
        
    def arrange(self):
        n_dims = np.array(results["trans_space.n_dimensions"])
        est_results = {"samples": {}, "estimates":{}}

        # For trans parameters
        for dim in np.unique(n_dims):
            dim = dim.item()  # This is unnescessary but it is just for GUI problem.
            est_results["samples"][dim] = {}
            idxs = np.where(n_dims == dim)[0]
            # Collect posterior samples
            for tn in trans_param_names:
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
    def estimate():
        return None
    def output():
        return None