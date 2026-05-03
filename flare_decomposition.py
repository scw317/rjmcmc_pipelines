from pathlib import Path

import bayesbay as bb
import numpy as np

from util import InversionRepeater

# ================================
# %% Data
# ++++++++++++++++++++++++++++++++

# Data path
data_path = Path.home() / "Dropbox/workspace/paper/Kang2026-sub/analysis/ovro.txt"
data = np.loadtxt(data_path, comments="#")

# Data save path
save_dir = Path.home() / "Dropbox/workspace/paper/Kang2026-sub/analysis/ovro/test2"

# ================================
# %% Sampling arguments preset
# ++++++++++++++++++++++++++++++++

# The limit number of model components
n_dim_min = 1
n_dim_max = 40

# The final number of posterior samples is
# (The number of T=1 chains) * (n_iterations - brunin_iterations) / save_every.
n_chains = 16
n_iterations = 2000000
burnin_iterations = 1600000
save_every = 1000
verbose = True
print_every = 100000

sampler = bb.samplers.ParallelTempering(temperature_max=5, swap_every=10000)
#sampler = bb.samplers.SimulatedAnnealing(temperature_start=10)

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

# ================================
# %% Prior preset
# ++++++++++++++++++++++++++++++++

period = data[-1, 0] - data[0, 0]
time_gaps = (np.roll(data[:, 0], -1) - data[:, 0])[:-1]
flux_gaps = np.abs((np.roll(data[:, 1], -1) - data[:, 1]))[:-1]

# Trans-diemnsional parameters limitation
trans_param_limits = [
    [data[:, 0].min(), data[:, 0].max()],
    [flux_gaps.min(), data[:, 1].max()],
    [time_gaps.min(), period],
    [0.1, 5],
]
# Fixed-dimensional parameters limitation
fixed_param_limits = [
    [0, data[:, 1].min()],
]

# Standard deviations of perturbation distribution
trans_param_perturb_stds = np.diff(np.array(trans_param_limits)).flatten() * 0.001
fixed_param_perturb_stds = np.diff(np.array(fixed_param_limits)).flatten() * 0.005

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

inversion_repeater = InversionRepeater(inversion, sort_refs=sort_refs, save_dir=save_dir)

inversion_repeater.run(
    sampler=sampler,
    n_iterations=n_iterations,
    burnin_iterations=burnin_iterations,
    save_every=save_every,
    verbose=verbose,
    print_every=print_every,
)

