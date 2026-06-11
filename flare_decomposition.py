import csv

import bayesbay as bb
import numpy as np
from numba import njit

from rjmcmc.handle import InversionHandler

# ================================
# %% Preset: Data
# ++++++++++++++++++++++++++++++++

data_path = "flux_data.csv"
data = np.loadtxt(data_path)

times = data[..., 0]
fluxes = data[..., 1]
errors = data[..., 2]

# Directory path to save results
save_dir = "./results"

# ================================
# %% Normalize data
# ++++++++++++++++++++++++++++++++

# Normalize times from -0.5 to 0.5
times_scale = times.max() - times.min()
times_offset = times.min() + times_scale / 2.0
times_n = (times - times_offset) / times_scale 

# Normalized flux densities and their errors
fluxes_scale = 1.4826 * np.median(np.abs(fluxes))
fluxes_offset = 0.0
fluxes_n = (fluxes - fluxes_offset) / fluxes_scale
errors_n = errors / fluxes_scale

# Normalizing constants
normalizing = {
    "times_scale": times_scale,
    "times_offset": times_offset,
    "fluxes_scale": fluxes_scale,
    "fluxes_offset": fluxes_offset,
}

# Write normalizing constants as CSV
with open(save_dir / "normalizing.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["key", "value"])
    for key, val in normalizing.items():
        writer.writerow([key, val])
        
# ================================
# %% Preset: Sampling arguments
# ++++++++++++++++++++++++++++++++

# The limit numbers of model components
min_dim = 1
max_dim = 20

# The number of Markov chains
n_chains = 8

# Keyword arguments for bayesbay.BayesianInversion.run()
run_kwargs = {
    "n_iterations": 200000,
    "burnin_iterations" : 100000,
    "save_every": 1000,
    "verbose": True,
    "print_every": 100000,
    #"sampler": bb.samplers.ParallelTempering(temperature_max=5.0, swap_every=10000),
    "sampler": bb.samplers.SimulatedAnnealing(temperature_start=10.0),
    #"sampler": bb.samplers.VanillaSampler()
}

# ================================
# %% Preset: Model parameters
# ++++++++++++++++++++++++++++++++

# Trans-dimensional parameter space
trans_space_name = "flare"
trans_param_names = ["t0", "f0", "tau", "s"]

# Fixed-dimensional parameters space
fixed_space_name = "quiescent"
fixed_param_names = ["fqs"]

# Reference parameter sorting samples
sort_refs = ["flare.t0"]

# Limits of Trans-diemnsional parameters
trans_limits = [
    [-0.75, 0.75],
    [3.0 * errors_n.min(), fluxes_n.max()],
    [0.001, 0.5],
    [0.1, 5.0],
]

# Limits of fixed-diemnsional parameters
fixed_limits = [
    [0, fluxes.min()],
]

# Standard deviations of perturbation distributions
trans_perturb_stds = [0.005 * (lim[1] - lim[0]) for lim in trans_limits]
fixed_perturb_stds = [0.05 * (lim[1] - lim[0]) for lim in fixed_limits]

# ================================
# %% Priors
# ++++++++++++++++++++++++++++++++

trans_priors = []
for name, lim, std in zip(trans_param_names, trans_limits, trans_perturb_stds):
    prior = bb.prior.UniformPrior(
        name=name, vmin=lim[0], vmax=lim[1], perturb_std=std,
    )
    trans_priors.append(prior)

fixed_priors = []
for name, lim, std in zip(fixed_param_names, fixed_limits, fixed_perturb_stds):
    prior = bb.prior.UniformPrior(
        name=name, vmin=lim[0], vmax=lim[1], perturb_std=std,
    )
    fixed_priors.append(prior)

# ================================
# %% Modeling
# ++++++++++++++++++++++++++++++++

@njit
def forward_func(t, t0, f0, tau, s, fqs, etol=64, atol=1e-10):
    # Forward model value memory covered with quiescent flux density
    f = np.full(len(t), fqs)

    # Sum over component model values
    for i in range(len(t0)):
        # Caching component model parameters
        _t0  = t0[i]          
        _f0  = f0[i]
        _tau = tau[i]
        _s   = s[i]
        for j in range(len(t)):
            x = (t[j] - _t0) / _tau
            if x < -etol:
                x = -etol
            elif x > etol:
                x = etol
            f[j] += 2.0 * _f0 / (np.exp(-x) + np.exp(x / _s) + atol)

    return f

# ================================
# %% Likelihood
# ++++++++++++++++++++++++++++++++

log_factor = -0.5 * np.sum(np.log(2.0 * np.pi * errors_n**2))


@njit
def log_likelihood_func(t, t0, f0, tau, s, fqs, fluxes, errors, log_factor):
    f = forward_func(t, t0, f0, tau, s, fqs)
    log_exp = 0.0
    for i in range(len(f)):
        log_exp += -0.5 * ((fluxes_n[i] - f[i]) / errors_n[i])**2
    return log_factor + log_exp


def log_like_func(state: bb.State):
    t0 = state["flare"]["t0"]
    f0 = state["flare"]["f0"]
    tau = state["flare"]["tau"]
    s = state["flare"]["s"]
    fqs = state["quiescent"]["fqs"][0]
    return log_likelihood_func(
        times_n, t0, f0, tau, s, fqs,
        fluxes_n, errors_n, log_factor
    )
    
    
log_likelihood = bb.likelihood.LogLikelihood(log_like_func=log_like_func)

# ================================
# %% Bayesbay instances
# ++++++++++++++++++++++++++++++++

trans_space = bb.parameterization.ParameterSpace(
    name=trans_space_name,
    n_dimensions=None,
    n_dimensions_min=min_dim,
    n_dimensions_max=max_dim,
    parameters=trans_priors,
)

fixed_space = bb.parameterization.ParameterSpace(
    name=fixed_space_name,
    n_dimensions=len(fixed_priors),
    parameters=fixed_priors,
)

parameterization = bb.parameterization.Parameterization([trans_space, fixed_space])
parameterization.initialize()

inversion = bb.BayesianInversion(
    parameterization=parameterization,
    log_likelihood=log_likelihood,
    n_chains=n_chains,
)

# ================================
# %% Run sampling
# ++++++++++++++++++++++++++++++++

save_dir.mkdir(parents=True, exist_ok=True)

handler = InversionHandler(inversion, sort_refs, save_dir)
handler.run(**run_kwargs)