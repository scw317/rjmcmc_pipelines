import csv
import pickle
from datetime import datetime
from inspect import signature
from pathlib import Path
from typing import Any, Sequence

import bayesbay as bb
import pandas as pd

from process import PostProcess
from util import expand_array_columns, organize_results

# Signature keyword arguments of bb.BayesianInversion.run()
SIGNATURE_BB_RUN_KEYS = list(signature(bb.BayesianInversion.run).parameters.keys())


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