"""
estimate_multi_edge_betas.py

Loops over several treatment → outcome pairs, uses DoWhy to find the minimal
back–door adjustment set for each pair, then fits a Bayesian single–break
hinge model (PyMC v5) to estimate:
    • β_below   slope when treatment ≤ θ
    • β_above   slope when treatment >  θ  (β_below + β_increment)
    • θ         changepoint (minutes / score units)
and 90% HDIs for each.

It writes one row per edge to LifestyleDAG_betas.csv.

Requires:
    pip install dowhy==0.11.0 pymc==5.10.0 arviz==0.16.1 pandas numpy scipy>=1.8
"""
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# stdlib / third-party imports
import pandas as pd
import numpy as np
import scipy.signal as _sig
import sys
import networkx as nx
import multiprocessing

# ------------------------------------------------------------------
# Compatibility shim: SciPy >=1.8 moved the Gaussian window helper
# from ``scipy.signal.gaussian`` to ``scipy.signal.windows.gaussian``.
# ArviZ<0.17 still expects the old location.  If it doesn't exist,
# create an alias before importing ArviZ / PyMC.
# ------------------------------------------------------------------
if not hasattr(_sig, "gaussian"):
    from scipy.signal.windows import gaussian as _gaussian  # type: ignore
    _sig.gaussian = _gaussian  # type: ignore[attr-defined]
    sys.modules['scipy.signal'].gaussian = _gaussian  # ensure alias for future imports

# Now the heavy libraries that rely on the alias.
import pymc as pm
import arviz as az
from dowhy import CausalModel

# ----------------------------------------------------------------------
DATA_FILE   = 'LifestyleDAG_norm.csv'
OUTPUT_FILE = 'LifestyleDAG_betas.csv'
HDI_PROB    = 0.90    # credible interval probability

# PyMC sampling parameters
N_DRAWS = 1000
N_TUNE = 1000
N_CHAINS = 4
N_CORES = 4
TARGET_ACCEPT = 0.95
RANDOM_SEED = 42

df = pd.read_csv(DATA_FILE)

# -- explicit graph edges (avoids pygraphviz / pydot) -----------------
GRAPH_EDGES = [
    ('precipitationFlag', 'zone2Minutes'),
    ('precipitationFlag', 'sleepQuality'),
    ('precipitationFlag', 'affectToday'),
    ('zone2Minutes', 'trainingFatigueScore'),
    ('zone2Minutes', 'sleepQuality'),
    ('zone2Minutes', 'affectToday'),
    ('trainingFatigueScore', 'sleepQuality'),
]

CAUSAL_GRAPH_NX = nx.DiGraph()
CAUSAL_GRAPH_NX.add_edges_from(GRAPH_EDGES)

# ----- continuous-treatment edge list ----------------------------------
EDGE_SPECS = [
    {'treatment': 'zone2Minutes',         'outcome': 'sleepQuality'},
    {'treatment': 'zone2Minutes',         'outcome': 'affectToday'},
    {'treatment': 'trainingFatigueScore', 'outcome': 'sleepQuality'},
]

# ---------------------------------------------------------------------
# EXTRA: binary-treatment edges to analyse via linear ATE
BINARY_EDGES = [
    {"treatment": "precipitationFlag", "outcome": "zone2Minutes"},
    {"treatment": "precipitationFlag", "outcome": "sleepQuality"},
    {"treatment": "precipitationFlag", "outcome": "affectToday"},
]
# ---------------------------------------------------------------------

# -- main execution -----------------------------------------------------
def run_estimation():
    # Load data
    df = pd.read_csv(DATA_FILE)
    raw_df = pd.read_csv("LifestyleDAG.csv")   # for back-transforming continuous results

    results = []

    # 1) Continuous-treatment hinge models  (unchanged) ........................
    for spec in EDGE_SPECS:
        t_name = spec["treatment"]
        y_name = spec["outcome"]
        print(f"\nEstimating edge: {t_name} -> {y_name}")

        # Initialize CausalModel with the NetworkX graph
        # Filter df to only include necessary columns for this specific model
        # to avoid issues with CausalModel trying to interpret all columns
        current_vars = {t_name, y_name}
        # Add potential confounders from the graph to current_vars
        # This is a simplified approach; a more robust way would be to get all nodes
        # or parse CAUSAL_GRAPH to find all relevant variable names.
        # For now, assume all columns in df *could* be relevant if not t_name or y_name.
        # However, CausalModel will use the graph to determine actual confounders.
        
        # Create a DiGraph from the edge list
        # We need all nodes that might be involved in any path related to any edge.
        # A simple way is to get all unique node names from GRAPH_EDGES.
        all_nodes_in_graph = set(n for e in GRAPH_EDGES for n in e)
        
        # Ensure all nodes used in EDGE_SPECS are also in all_nodes_in_graph,
        # or at least that CausalModel can handle nodes present in data but not explicitly in graph if that's intended.
        # For CausalModel, it's usually best if the graph covers all variables in the dataframe subset passed to it.
        
        # Let's refine the graph passed to CausalModel.
        # It should contain all variables in the dataframe that could be relevant.
        # For simplicity, we're building the full graph each time.
        # If performance were an issue for very large graphs/many edges, one could optimize.
        
        g = nx.DiGraph()
        g.add_edges_from(GRAPH_EDGES)
        # Add isolated nodes if they are in the data and might be relevant (e.g. as treatments/outcomes not in edges)
        # For this script, GRAPH_EDGES should define the full causal structure being considered.
        # Nodes present in data but not in graph are handled by DoWhy depending on context.

        cm = CausalModel(data=df, treatment=t_name, outcome=y_name, graph=g)


        # Identify estimand
        est = cm.identify_effect(proceed_when_unidentifiable=True)
        print(f"Estimand for {t_name} -> {y_name}: {est}")

        # Store confounders used
        confounders_used = est.get_backdoor_variables()
        print(f"  Back-door set: {confounders_used}")


        # Fit PyMC model using the identified confounders
        with pm.Model() as pymc_model:
            # Data
            x_data = pm.MutableData("x_data", df[t_name].values)
            y_data = pm.MutableData("y_data", df[y_name].values)
            conf_data = pm.MutableData("conf_data", df[confounders_used].values if confounders_used else np.array([]).reshape(len(df),0) )

            # Priors
            alpha = pm.Normal("alpha", mu=0, sigma=1)
            beta1 = pm.Normal("beta1", mu=0, sigma=0.2)  # Tighter prior
            beta2_offset = pm.Normal("beta2_offset", mu=0, sigma=0.2) # Tighter prior
            beta2 = pm.Deterministic("beta2", beta1 + beta2_offset) # Actual slope after

            # Changepoint must be within observed range of treatment
            theta_lower = df[t_name].min()
            theta_upper = df[t_name].max()
            theta = pm.Uniform("theta", lower=theta_lower, upper=theta_upper)

            sigma = pm.HalfNormal("sigma", sigma=1)

            # Confounder effects (if any)
            if confounders_used:
                gammas = pm.Normal("gammas", mu=0, sigma=0.5, shape=len(confounders_used))
                conf_effect = pm.math.dot(conf_data, gammas)
            else:
                conf_effect = 0.0

            # Linear model with changepoint
            # mu = alpha + beta1 * x_data * (x_data <= theta) + beta2 * (x_data - theta) * (x_data > theta) + conf_effect
            # Simpler way to express hinge:
            idx = x_data > theta
            mu_ = alpha + beta1 * x_data + beta2_offset * pm.math.switch(idx, x_data - theta, 0) + conf_effect

            # Likelihood
            likelihood = pm.Normal("likelihood", mu=mu_, sigma=sigma, observed=y_data)

            # Sampling
            trace = pm.sample(
                N_DRAWS,
                tune=N_TUNE,
                chains=N_CHAINS,
                cores=N_CORES,
                target_accept=TARGET_ACCEPT,
                random_seed=RANDOM_SEED,
            )

        # Summarize results
        summary = az.summary(trace, var_names=["beta1", "beta2", "theta"], hdi_prob=HDI_PROB)
        hdi_cols = sorted([c for c in summary.columns if c.startswith("hdi_")])
        if len(hdi_cols) != 2:
            raise ValueError("Unexpected HDI columns returned by ArviZ: " + str(hdi_cols))
        lower_hdi, upper_hdi = hdi_cols

        # For back-transforming to raw units, load means/stds from the original (unnormalized) data
        x_mean = raw_df[t_name].mean()
        x_std = raw_df[t_name].std()
        y_mean = raw_df[y_name].mean()
        y_std = raw_df[y_name].std()

        # Extract normalized results
        beta1_mean = summary.loc["beta1", "mean"]
        beta1_low = summary.loc["beta1", lower_hdi]
        beta1_high = summary.loc["beta1", upper_hdi]
        beta2_mean = summary.loc["beta2", "mean"]
        beta2_low = summary.loc["beta2", lower_hdi]
        beta2_high = summary.loc["beta2", upper_hdi]
        theta_mean = summary.loc["theta", "mean"]
        theta_low = summary.loc["theta", lower_hdi]
        theta_high = summary.loc["theta", upper_hdi]

        # Back-transform to raw units:
        # For normalized data: y = (raw_y - y_mean) / y_std, x = (raw_x - x_mean) / x_std
        # The regression slope in raw units: beta_raw = beta_norm * (y_std / x_std)
        # The changepoint in raw units: theta_raw = theta_norm * x_std + x_mean
        beta1_mean_raw = beta1_mean * (y_std / x_std)
        beta1_low_raw = beta1_low * (y_std / x_std)
        beta1_high_raw = beta1_high * (y_std / x_std)
        beta2_mean_raw = beta2_mean * (y_std / x_std)
        beta2_low_raw = beta2_low * (y_std / x_std)
        beta2_high_raw = beta2_high * (y_std / x_std)
        theta_mean_raw = theta_mean * x_std + x_mean
        theta_low_raw = theta_low * x_std + x_mean
        theta_high_raw = theta_high * x_std + x_mean

        results.append(
            {
                "edge": f"{t_name}_to_{y_name}",
                # Normalized units
                "beta_below_mean_norm": beta1_mean,
                "beta_below_hdi_low_norm": beta1_low,
                "beta_below_hdi_high_norm": beta1_high,
                "beta_above_mean_norm": beta2_mean,
                "beta_above_hdi_low_norm": beta2_low,
                "beta_above_hdi_high_norm": beta2_high,
                "theta_mean_norm": theta_mean,
                "theta_hdi_low_norm": theta_low,
                "theta_hdi_high_norm": theta_high,
                # Raw units
                "beta_below_mean_raw": beta1_mean_raw,
                "beta_below_hdi_low_raw": beta1_low_raw,
                "beta_below_hdi_high_raw": beta1_high_raw,
                "beta_above_mean_raw": beta2_mean_raw,
                "beta_above_hdi_low_raw": beta2_low_raw,
                "beta_above_hdi_high_raw": beta2_high_raw,
                "theta_mean_raw": theta_mean_raw,
                "theta_hdi_low_raw": theta_low_raw,
                "theta_hdi_high_raw": theta_high_raw,
                "confounders_used": ", ".join(confounders_used) if confounders_used else "None",
            }
        )
    # -------------------------------------------------------------------------

    # 2) Binary-treatment ATEs ................................................
    for spec in BINARY_EDGES:
        t_name, y_name = spec["treatment"], spec["outcome"]
        print(f"\nEstimating ATE: {t_name} -> {y_name}")

        cm = CausalModel(data=df, treatment=t_name, outcome=y_name, graph=CAUSAL_GRAPH_NX)
        estimand = cm.identify_effect(proceed_when_unidentifiable=True)
        backdoor = estimand.get_backdoor_variables()

        estimator = cm.estimate_effect(
            estimand,
            method_name="backdoor.linear_regression",
        )
        ate, ate_ci = estimator.value, estimator.get_confidence_intervals()

        results.append({
            "edge": f"{t_name}_to_{y_name}",
            "effect_type": "ATE",
            "ate_mean": ate,
            "ate_ci_low": ate_ci[0][0],
            "ate_ci_high": ate_ci[0][1],
            "confounders_used": ", ".join(backdoor) if backdoor else "None",
        })
    # -------------------------------------------------------------------------

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved results to {OUTPUT_FILE}")

if __name__ == '__main__':
    multiprocessing.freeze_support() # Added for Windows multiprocessing
    run_estimation() # Call the main logic
