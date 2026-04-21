from pathlib import Path

import bayesbay as bb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util import PostResult

# ================================
# %% Data
# ++++++++++++++++++++++++++++++++

# Data path
data_path = Path.home() / "Dropbox/workspace/paper/Kang2026-sub/analysis/bu.txt"
data = np.loadtxt(data_path, comments="#")

# Data save path
save_dir = Path.home() / "Dropbox/workspace/paper/Kang2026-sub/analysis/bu/test2"
save_dir.mkdir(parents=True, exist_ok=True)

# ================================
# %% Sampling arguments preset
# ++++++++++++++++++++++++++++++++

# The limit number of model components
n_dim_min = 1
n_dim_max = 10

# The final number of posterior samples is
# (The number of T=1 chains) * (n_iterations - brunin_iterations) / save_every.
n_chains = 32
n_iterations = 10000
burnin_iterations = 1000
save_every = 1000
verbose = True
print_every = 10000

# ================================
# %% Parameters preset
# ++++++++++++++++++++++++++++++++

# Trans models are sorted by order of "sort_ref".
sort_ref = "t0"

# Trans parameters
trans_param_names = ["t0", "f0", "tau", "s"]
# Fixed parameters
fixed_param_names = ["fqs"]

period = data[-1, 0] - data[0, 0]
time_gaps = (np.roll(data[:, 0], -1) - data[:, 0])[:-1]
flux_gaps = (np.roll(data[:, 1], -1) - data[:, 1])[:-1]

trans_param_limits = [
    [data[:, 0].min(), data[:, 0].max()],
    [0, data[:, 1].max()],
    [time_gaps.min(), period],
    [0.1, 5],
]

fixed_param_limits = [
    [0, data[:, 1].min()],
]

# ================================
# %% Prior preset
# ++++++++++++++++++++++++++++++++

# Relative stadard deviation to limit range for perturbation distribution.
# The lower value, the more precies but longer sampling is proceeded.
rel_std = 0.01

trans_priors = []
for name, lim in zip(trans_param_names, trans_param_limits):
    prior = bb.prior.UniformPrior(
        name=name,
        vmin=lim[0],
        vmax=lim[1],
        perturb_std=(lim[1] - lim[0])*rel_std,
    )
    trans_priors.append(prior)

fixed_priors = []
for name, lim in zip(fixed_param_names, fixed_param_limits):
    prior = bb.prior.UniformPrior(
        name=name,
        vmin=lim[0],
        vmax=lim[1],
        perturb_std=(lim[1] - lim[0])*rel_std,
    )
    fixed_priors.append(prior)

# ================================
# %% Parameterization
# ++++++++++++++++++++++++++++++++

trans_space = bb.parameterization.ParameterSpace(
    name="trans_space",
    n_dimensions=None,
    n_dimensions_min=n_dim_min,
    n_dimensions_max=n_dim_max,
    parameters=trans_priors,
)

fixed_space = bb.parameterization.ParameterSpace(
    name="fixed_space",
    n_dimensions=len(fixed_priors),
    parameters=fixed_priors,
)

parameterization = bb.parameterization.Parameterization([trans_space, fixed_space])
parameterization.initialize()

# ================================
# %% Modeling
# ++++++++++++++++++++++++++++++++

def model_func(t, t0, f0, tau, s):
    x = np.clip((t - t0) / tau, -100, 100) # Clipping to prevent overflow
    return 2 * f0 / (np.exp(-x) + np.exp(x) + 1e-16)


def fwd_func(state):
    t0_arr = state["trans_space"]["t0"][..., np.newaxis]
    f0_arr = state["trans_space"]["f0"][..., np.newaxis]
    tau_arr = state["trans_space"]["tau"][..., np.newaxis]
    s_arr = state["trans_space"]["s"][..., np.newaxis]
    fqs = state["fixed_space"]["fqs"]
    
    if t0_arr.size == 0:
        return np.full(data[:, 0].shape, fqs)
    
    model_vals = model_func(data[:, 0], t0_arr, f0_arr, tau_arr, s_arr)
    fwd_vals = np.sum(model_vals, axis=0) + fqs
        
    return fwd_vals

# ================================
# %% Likelihood
# ++++++++++++++++++++++++++++++++

target = bb.likelihood.Target(
    name="flux_density",
    dobs=data[:, 1],
    covariance_mat_inv=1/data[:, 2]**2,
)

log_likelihood = bb.likelihood.LogLikelihood(
    targets=[target],
    fwd_functions=[fwd_func],
    )

# ================================
# %% Sampling
# ++++++++++++++++++++++++++++++++

inversion = bb.BayesianInversion(
    parameterization=parameterization,
    log_likelihood=log_likelihood,
    n_chains=n_chains,
)

inversion.run(
    sampler=bb.samplers.SimulatedAnnealing(),
    n_iterations=n_iterations,
    burnin_iterations=burnin_iterations,
    save_every=save_every,
    verbose=verbose,
    print_every=print_every,
)

for chain in inversion.chains:
    chain.print_statistics()
results = inversion.get_results(concatenate_chains=False)

result_df = pd.DataFrame(results)
result_df.to_parquet(
    path=save_dir / "results.parquet",
    engine="pyarrow", compression="zstd"
)

post_result = PostResult(result_df, False)

'''
# ================================
# %% Parameter estimation
# ++++++++++++++++++++++++++++++++

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

# ================================
# %% Export samples to txt
# ++++++++++++++++++++++++++++++++

for dim in np.unique(n_dims):
    for tn in trans_param_names:
        np.savetxt(save_dir / f"trans_{dim}.txt", est_results["samples"][dim][tn])
    np.savetxt(save_dir / f"fwd_{dim}.txt", est_results["samples"][dim]["forward"])

for fn in fixed_param_names:
    np.savetxt(save_dir / "fixed.txt", est_results["samples"][fn])

# ================================
# %% Export estimates to excel
# ++++++++++++++++++++++++++++++++

with pd.ExcelWriter(save_dir / "results.xlsx") as w:
    # Dimension sheet
    dim_df = pd.DataFrame(data=np.vstack(np.unique(n_dims, return_counts=True)).T, columns=["dim", "num"])
    dim_df.to_excel(w, sheet_name="dim", index=False)
    # Fixed parameter sheet
    est_type_col_df = pd.DataFrame(data=["mean", "median", "lower", "upper"], columns=["est_type"])
    param_col_df = pd.DataFrame(
        data=np.hstack([est_results["estimates"][fn][..., np.newaxis] for fn in fixed_param_names]),
        columns=fixed_param_names,
    )
    fixed_df = pd.concat((est_type_col_df, param_col_df), axis=1)
    fixed_df.to_excel(w, sheet_name="fixed", index=False)
    # Trans parameter sheet
    trand_df_ls = []
    for dim in np.unique(n_dims):
        dim_col_df = pd.DataFrame(np.repeat(dim, dim * 4), columns=["dim"])
        seq_col_df = pd.DataFrame(np.tile(np.arange(dim), 4), columns=["seq"])
        est_type_col_df = pd.DataFrame(np.repeat(["mean", "median", "lower", "upper"], dim), columns=["est_type"])
        param_col_df = pd.DataFrame(
            data=np.hstack([np.hstack(est_results["estimates"][dim][tn])[..., np.newaxis] for tn in trans_param_names]),
            columns=trans_param_names,
        )
        trans_df = pd.concat((dim_col_df, est_type_col_df, seq_col_df, param_col_df), axis=1)
        trand_df_ls.append(trans_df)
    all_trans_df = pd.concat(trand_df_ls, axis=0)
    all_trans_df.to_excel(w, sheet_name="trans", index=False)

# ================================
# %% Plot dimension histogram
# ++++++++++++++++++++++++++++++++

fig, ax = plt.subplots(dpi=300)
ax.hist(n_dims, bins=np.arange(n_dims.min(), n_dims.max() + 2) - 0.5)
ax.set_xlabel("dim")
ax.set_ylabel("# of posterior samples")
ax.set_title("Posterior distribution of dimensions")
fig.savefig(save_dir / "dim.png")

# ================================
# %% Plot fixed parameter histogram
# ++++++++++++++++++++++++++++++++

fig, axes = plt.subplots(dpi=300, ncols=len(fixed_param_names))
for f, fn in enumerate(fixed_param_names):
    axes = np.atleast_1d(axes)
    axes[f].hist(est_results["samples"][fn])
    axes[f].set_xlabel(fn)
fig.supylabel("# of posterior samples")
fig.suptitle("Posterior distribution of fixed parameters")
fig.savefig(save_dir / "fixed.png")

# ================================
# %% Plot trans parameter histogram
# ++++++++++++++++++++++++++++++++

for dim in np.unique(n_dims):
    fig, axes = plt.subplots(dpi=300, ncols=len(trans_param_names), nrows=dim, figsize=(2*len(trans_param_names), 2*dim))
    axes = np.atleast_2d(axes)
    for t, tn in enumerate(trans_param_names):
        for s, sam in enumerate(est_results["samples"][dim][tn].T):
            axes[s, t].hist(sam)
            axes[s, 0].set_ylabel(f"seq_{s}")
        axes[s, t].set_xlabel(f"{tn}")
    fig.supylabel("# of posterior samples")
    fig.suptitle(f"Posterior distribution of trans parameters (dim={dim})")
    fig.tight_layout()
    fig.savefig(save_dir / f"trans_{dim}.png", bbox_inches="tight")

# ================================
# %% Plot model
# ++++++++++++++++++++++++++++++++

plot_data = np.linspace(data[:, 0].min(), data[:, 0].max(), 1000)

for dim in np.unique(n_dims):    
    fig, ax = plt.subplots(dpi=300)
    ax.errorbar(data[:, 0], data[:, 1], data[:, 2], ls="none", marker=".", color="black", label="Data")
    mean, median, lower, upper = est_results["estimates"][dim]["forward"]
    ax.plot(data[:, 0], mean, lw=1.5, color="red", label="Mean forward")
    ax.plot(data[:, 0], median, lw=1, color="orange", label="Median forward")
    ax.fill_between(data[:, 0], lower, upper, color="red", alpha=0.3, label=r"$1\sigma$ CI")
    model_val = model_func(
        plot_data,
        *[est_results["estimates"][dim][tn][0][..., np.newaxis] for t, tn in enumerate(trans_param_names)],
    ) + est_results["estimates"]["fqs"][0]
    ax.plot(plot_data, model_val.T, lw=0.5, color="violet")
    ax.legend()
    ax.set_xlabel("MJD")
    ax.set_ylabel("Flux density [Jy]")
    ax.set_title(f"Data and model plot (dim={dim})")
    fig.savefig(save_dir / f"model_{dim}.png")
'''