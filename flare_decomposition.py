from pathlib import Path

import bayesbay as bb
import numpy as np

# ================================
# %% Data
# ++++++++++++++++++++++++++++++++

# Data path
data_path = Path.home() / "Dropbox/workspace/paper/Kang2026-sub/analysis/bu.txt"
data = np.loadtxt(data_path, comments="#")

# Data save path
save_dir = Path.home() / "Dropbox/workspace/paper/Kang2026-sub/analysis/bu/test13"

# ================================
# %% Sampling arguments preset
# ++++++++++++++++++++++++++++++++

# The limit number of model components
n_dim_min = 10
n_dim_max = 30

# The final number of posterior samples is
# (The number of T=1 chains) * (n_iterations - brunin_iterations) / save_every.
n_chains = 20
n_iterations = 2000000
burnin_iterations = 1000000
save_every = 1000
verbose = True
print_every = 500000

#sampler = bb.samplers.SimulatedAnnealing(temperature_start=20)
sampler = bb.samplers.ParallelTempering(temperature_max=5, swap_every=1000)

# ================================
# %% Parameters preset
# ++++++++++++++++++++++++++++++++

# Trans-dimensional parameter space
trans_space_name = "flare"
trans_param_names = ["t0", "f0", "tau", "s"]

# Fixed-dimensional parameters space
fixed_space_name = "quiescent"
fixed_param_names = ["fqs"]

# Sort samples refered by sort_ref
sort_refs = ["flare.t0"]

period = data[-1, 0] - data[0, 0]
time_gaps = (np.roll(data[:, 0], -1) - data[:, 0])[:-1]
flux_gaps = np.abs((np.roll(data[:, 1], -1) - data[:, 1]))[:-1]

# Parameters limitation
trans_param_limits = [
    [data[:, 0].min(), data[:, 0].max()],
    [flux_gaps.min(), data[:, 1].max()],
    [time_gaps.min(), period],
    [0.1, 5],
]
fixed_param_limits = [
    [0, data[:, 1].min()],
]

# Standard deviations of perturbation distribution
trans_param_perturb_stds = np.diff(np.array(trans_param_limits)).flatten() * 0.005
fixed_param_perturb_stds = np.diff(np.array(fixed_param_limits)).flatten() * 0.1

# ================================
# %% Prior preset
# ++++++++++++++++++++++++++++++++

trans_priors = []
for name, lim, std in zip(trans_param_names, trans_param_limits, trans_param_perturb_stds):
    prior = bb.prior.UniformPrior(
        name=name,
        vmin=lim[0],
        vmax=lim[1],
        perturb_std=std,
    )
    trans_priors.append(prior)

fixed_priors = []
for name, lim, std in zip(fixed_param_names, fixed_param_limits, fixed_param_perturb_stds):
    prior = bb.prior.UniformPrior(
        name=name,
        vmin=lim[0],
        vmax=lim[1],
        perturb_std=std,
    )
    fixed_priors.append(prior)

# ================================
# %% Parameterization
# ++++++++++++++++++++++++++++++++

trans_space = bb.parameterization.ParameterSpace(
    name=trans_space_name,
    n_dimensions=None,
    n_dimensions_min=n_dim_min,
    n_dimensions_max=n_dim_max,
    parameters=trans_priors,
)

fixed_space = bb.parameterization.ParameterSpace(
    name=fixed_space_name,
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
    t0_arr = state[trans_space_name]["t0"][..., np.newaxis]
    f0_arr = state[trans_space_name]["f0"][..., np.newaxis]
    tau_arr = state[trans_space_name]["tau"][..., np.newaxis]
    s_arr = state[trans_space_name]["s"][..., np.newaxis]
    fqs = state[fixed_space_name]["fqs"]
    
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
    sampler=sampler,
    n_iterations=n_iterations,
    burnin_iterations=burnin_iterations,
    save_every=save_every,
    verbose=verbose,
    print_every=print_every,
)


'''
# ================================
# %% Plot model
# ++++++++++++++++++++++++++++++++

data_path = save_dir / 

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