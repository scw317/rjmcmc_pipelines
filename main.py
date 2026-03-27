from pathlib import Path

import bayesbay as bb
import matplotlib.pyplot as plt
import numpy as np

# ================================
# %% Import data
# ++++++++++++++++++++++++++++++++

data_path = Path.home() / "Dropbox/workspace/data/bu.txt"
data = np.loadtxt(data_path, comments="#")

# ================================
# %% Sampling arguments
# ++++++++++++++++++++++++++++++++

n_chains = 50
n_iterations = 100000
burnin_iterations = 10000 
save_every = 100
verbose = False

# ================================
# %% Parameter preset
# ++++++++++++++++++++++++++++++++

trans_param_names = ["t0", "f0", "tau", "s"]
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
    n_dimensions_min=1,
    n_dimensions_max=int((len(data) - len(fixed_priors) - 1)/len(trans_priors)),
    parameters=trans_priors,
)

fixed_space = bb.parameterization.ParameterSpace(
    name="fixed_space",
    n_dimensions=len(fixed_priors),
    parameters=fixed_priors,
)

parameterization=bb.parameterization.Parameterization([trans_space, fixed_space])
state = parameterization.initialize()

# ================================
# %% Modeling
# ++++++++++++++++++++++++++++++++

def model_func(t, t0, F0, tau, s):
    x = (t - t0) / tau
    return 2 * F0 / (np.exp(-x) + np.exp(x/s) + 1e-8)


def fwd_func(state):
    t0_arr = state["trans_space"]["t0"][..., np.newaxis]
    f0_arr = state["trans_space"]["f0"][..., np.newaxis]
    tau_arr = state["trans_space"]["tau"][..., np.newaxis]
    s_arr = state["trans_space"]["s"][..., np.newaxis]
    fqs = state["fixed_space"]["fqs"][..., np.newaxis]
    
    model_vals = model_func(data[:, 0], t0_arr, f0_arr, tau_arr, s_arr)
    fwd_vals = np.sum(np.sum(model_vals, axis=0) + fqs)
        
    return fwd_vals

# ================================
# %% Likelihood
# ++++++++++++++++++++++++++++++++

target = bb.likelihood.Target("target", dobs=data[:, 1], covariance_mat_inv=1/data[:, 2]**2)
log_likelihood = bb.likelihood.LogLikelihood(targets=target, fwd_functions=fwd_func)

# ================================
# %% Posterior sampling
# ++++++++++++++++++++++++++++++++

inversion = bb.BayesianInversion(
    parameterization, log_likelihood, n_chains=n_chains
    )

samples = inversion.run(
    n_iterations=n_iterations,
    burnin_iterations=burnin_iterations,
    save_every=save_every,
    verbose=verbose,
    )

for chain in inversion.chains:
    chain.print_statistics()
results = inversion.get_results(concatenate_chains=True)
