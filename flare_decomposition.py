import csv
from pathlib import Path

import bayesbay as bb
import numpy as np

from util import InversionHandler

# ================================
# %% Preset
# ++++++++++++++++++++++++++++++++

# Data path
data_path = Path.home() / "Dropbox/workspace/paper/Kang2026-sub/analysis/bu.txt"
data = np.loadtxt(data_path, comments="#")

# Results save path
save_dir = Path.home() / "Dropbox/workspace/paper/Kang2026-sub/analysis/bu/error_1_scaled"
save_dir.mkdir(parents=True, exist_ok=True)

domain = data[:, 0]
values = data[:, 1]
errors = data[:, 2]

# ================================
# %% Normalize data
# ++++++++++++++++++++++++++++++++

# Normalize domain from -0.5 to 0.5
domain_scale = (domain.max() - domain.min())
domain_offset = domain.min() + domain_scale / 2
domain_normalized = (domain - domain_offset) / domain_scale 

# Normalize values and errors by rms
values_scale = np.sqrt(np.mean(values**2))
values_offset = 0
values_normalized = (values - values_offset) / values_scale
errors_normalized = errors / values_scale

normalize_info = {
    "domain_scale": domain_scale,
    "domain_offset": domain_offset,
    "values_scale": values_scale,
    "values_offset": values_offset,
}

with open(save_dir / "normalize_info.csv", mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for key, value in normalize_info.items():
        writer.writerow([key, value])

# ================================
# %% Sampling arguments preset
# ++++++++++++++++++++++++++++++++

# The limit number of model components
n_dim_min = 1
n_dim_max = 40

# The final number of posterior samples is
# (The number of T=1 chains) * (n_iterations - brunin_iterations) / save_every.
n_chains = 24
n_iterations = 1000000
burnin_iterations = 900000
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

# Trans-diemnsional parameters limitation
trans_param_limits = [
    [-1, 1],
    [errors_normalized.min(), values_normalized.max()],
    [0.01, 0.5],
    [0.1, 5],
]
# Fixed-dimensional parameters limitation
fixed_param_limits = [
    [0, values_normalized.min()],
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
    x = np.clip((t - t0) / tau, -2**6, 2**6) # Clipping to prevent overflow
    return 2 * f0 / (np.exp(-x) + np.exp(x / s) + 2**-48)
    

def fwd_func(state):
    t0_arr = state[trans_space_name]["t0"][..., None]
    f0_arr = state[trans_space_name]["f0"][..., None]
    tau_arr = state[trans_space_name]["tau"][..., None]
    s_arr = state[trans_space_name]["s"][..., None]
    fqs = state[fixed_space_name]["fqs"]
    
    if t0_arr.size == 0:
        return np.full(data[:, 0].shape, fqs)
    
    model_vals = model_func(domain_normalized, t0_arr, f0_arr, tau_arr, s_arr)
    fwd_vals = np.sum(model_vals, axis=0) + fqs
        
    return fwd_vals

# ================================
# %% Likelihood
# ++++++++++++++++++++++++++++++++

target = bb.likelihood.Target(
    name="flux_density",
    dobs=values_normalized,
    covariance_mat_inv=1/errors_normalized**2,
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

inversion_handler = InversionHandler(inversion, sort_refs=sort_refs, save_dir=save_dir)

inversion_handler.run(
    sampler=sampler,
    n_iterations=n_iterations,
    burnin_iterations=burnin_iterations,
    save_every=save_every,
    verbose=verbose,
    print_every=print_every,
)

