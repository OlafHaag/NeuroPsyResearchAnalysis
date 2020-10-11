"""
Having a 3 x (3) Mixed Design: 3 conditions as BETWEEN factor, 3 blocks as WITHIN factor.
There are multiple measurements in each block (trials).
There are 2 dependent variables: parallel & orthogonal
Since 'parallel' and 'orthogonal' measures have the same unit, 
it can be viewed as 1 outcome called 'projection' in 2 different directions.

The interest here is in the VARIABILITY (e.g. standard deviation or variance) of the data 
for parallel & orthogonal outcomes for blocks 1, 2, and 3.
The analysis should enable to make statements about whether conditions, blocks, direction 
have an effect on the outcome.

Bonus:
Each user has an individual level of experience in the task.
Users rate the difficulty of each block they performed.

Attribution:
Some code snippets were taken from Statistical Rethinking with Python and PyMC3.
Statistical Rethinking with Python and PyMC3 by All Contributors is licensed under a 
Creative Commons Attribution 4.0 International License.
"""
# %%
import sys
import logging
from pathlib import Path
import pickle

import arviz as az
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import patsy
import pymc3 as pm
import scipy.stats as st
import seaborn as sns
import theano.tensor as tt
import theano

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

plt.style.use(['seaborn-colorblind'])

RANDOM_SEED = 4096
np.random.seed(180)

# %%
def Gauss2d(mu, cov, ci, ax=None):
    """Copied from statsmodel"""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    v_, w = np.linalg.eigh(cov)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan(u[1]/u[0])
    angle = 180 * angle / np.pi # convert to degrees

    for level in ci:
        v = 2 * np.sqrt(v_ * st.chi2.ppf(level, 2)) #get size corresponding to level
        ell = Ellipse(mu[:2], v[0], v[1], 180 + angle, facecolor='None',
                      edgecolor='k',
                      alpha=(1-level)*.5,
                      lw=1.5)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
    
    return ax

# %%
# Common folder paths.
data_path = Path.cwd() / 'data'
reports_path = Path.cwd() / 'reports'
figures_path = reports_path / 'figures'

# %%
# Load data.
trial_data = pd.read_csv(data_path / 'preprocessed/projections.csv', index_col='id')
users_raw = pd.read_csv(data_path / 'raw/users.csv')
blocks_raw = pd.read_csv(data_path / 'raw/blocks.csv', index_col='id')
# We only analyze the first session of each participant.
# Since we do partial pooling in the model, we keep the users that were previously excluded from analyses because they 
# had too few data points.
df = trial_data.loc[trial_data['session'] == 1].drop(['session', 'exclude', 'trial'], axis='columns')
df['rating'] = df['block_id'].map(blocks_raw['rating'])
df['gaming_exp'] = df['user'].map(users_raw['gaming_exp'])
# Our predictors are categories.
df[['user', 'condition', 'block_id', 'block', 'direction']] = df[['user', 'condition', 'block_id', 'block', 'direction']
                                                                ].astype('category')
# Order categories.
block_order = [1, 2, 3]
df['block'].cat.reorder_categories(block_order, inplace=True)
condition_order = ['df1', 'df2', 'df1|df2']
df['condition'].cat.reorder_categories(condition_order, inplace=True)

# %%
# Transform data. I'm interested in the variance for each direction.
# The mean of the squared projection is that block's variance.
# Transform squared data once more, so we can model it using a normal distribution to make things easier.
fitted_projection, fitted_lambda, boxcox_ci = st.boxcox(df['projection'].transform('square'), alpha=0.05)
df['bc_sq_projection'] = fitted_projection

# %%
# Vizualize the data.
g = sns.FacetGrid(df, col='block', row='user', hue="direction")
g.despine(offset=10)
g = (g.map(sns.distplot, "bc_sq_projection").add_legend(title="Projection", loc='upper right'))
for ax in g.axes.flat:
    ax.set_ylabel("Probability Density")
    ax.set_xlabel("BoxCox(Projection Length^2)")
plt.tight_layout()

fig_filepath = figures_path / 'histogram-projections.pdf'
plt.savefig(str(fig_filepath))
logging.info(f"Written figure to {fig_filepath.resolve()}")

# %%
# Coordinates.
conditions = df['condition'].cat.categories.tolist()
n_conditions = df['condition'].nunique()
condition_indices = df['condition'].cat.codes.values # Condition index for every single value.
condition_lookup = pd.Series(*pd.factorize(df['condition'].cat.categories), name='condition_idx_lookup')

# We need to be able to index by user. But the user IDs of the sample don't start at 0.
# We create an index and a mapping between them.
users = df['user'].cat.categories.tolist()
n_users = df['user'].nunique()
user_indices = df['user'].cat.codes.values  # User index for every single value.
user_lookup = pd.Series(*pd.factorize(df['user'].cat.categories), name='user_idx_lookup')
# User level predictors. More options: age group, gender.
# gaming_exp is not on an interval scale (no equidistance between items), but ordinal.
experience = df.drop_duplicates(subset="user")['gaming_exp']  # Shape = user_indices

# Block indices for 1,2,3.
blocks = df['block'].cat.categories.tolist()
n_blocks = df['block'].nunique()  # Same as using cat.categories.size
block_indices = df['block'].cat.codes.values # Block index for every single value.
block_lookup = pd.Series(*pd.factorize(df['block'].cat.categories), name='block_idx_lookup')
# Block ID indices. Shape = n_users * n_blocks.
block_ids = df['block_id'].cat.categories.tolist()
n_block_ids = df['block_id'].nunique()
block_id_indices = df['block_id'].cat.codes.values # Block index for every single value.
block_id_lookup = pd.Series(*pd.factorize(df['block_id'].cat.categories), name='block_id_idx_lookup')
# Block level predictors.
# Note: Ratings are correlated with block index.
ratings = df.drop_duplicates(subset=["user", 'block'])['rating']  # Shape = block_id_indices

# Coded direction of projection: 0 = orthogonal, 1 = parallel.
directions = df['direction'].cat.categories.tolist()
n_directions = df['direction'].nunique()
direction_indices = df['direction'].cat.codes.values
direction_lookup = pd.Series(*pd.factorize(df['direction'].cat.categories), name='direction_idx_lookup')

coords = {'Direction': directions,
          'Condition': conditions,
          'User' :users,
          'Block': blocks,
          'Block_ID': block_ids,
          'obs_id': np.arange(direction_indices.size)}

# %%
# Mulit-Level Model
# A frequentist HLM indicated that condition does not have an effect, so we don't implement that as a level here.
# But honestly, because I don't know how.
with pm.Model(coords=coords) as model:  # block, user, direction. Based on m14_3 Statistical Rethinking 2.
    user_idx = pm.Data("user_idx", user_indices, dims="obs_id")
    block_idx = pm.Data("block_idx", block_indices, dims="obs_id")
    direction_idx = pm.Data("direction_idx", direction_indices, dims="obs_id")
    
    # Fixed priors.
    g = pm.Normal("g", mu=0.0, sd=1.0, dims='Direction')
    sd_dist = pm.Exponential.dist(1.0)
    chol_user, _, _ = pm.LKJCholeskyCov(
        "chol_user", n=n_directions, eta=4, sd_dist=sd_dist, compute_corr=True
    )
    chol_block, _, _ = pm.LKJCholeskyCov(
        "chol_block", n=n_directions, eta=4, sd_dist=sd_dist, compute_corr=True
    )

    # Adaptive priors, non-centered.
    z_user = pm.Normal("z_user", 0.0, 1.0, dims=('Direction', 'User'))
    alpha = pm.Deterministic("alpha", pm.math.dot(chol_user, z_user))
    z_block = pm.Normal("z_block", 0.0, 1.0, dims=('Direction', 'Block'))
    beta = pm.Deterministic("beta", pm.math.dot(chol_block, z_block))

    theta = pm.Deterministic("theta", 
                             g[direction_idx] + alpha[direction_idx, user_idx] + beta[direction_idx, block_idx])
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    projection_data = pm.Data("projection_data", df['bc_sq_projection'], dims='obs_id')
    projection = pm.Normal("projection", mu=theta, sigma=sigma, observed=projection_data, dims="obs_id")

# %%
# Vizualize model.
graph = pm.model_to_graphviz(model)
graph.render(figures_path / 'HLM2', format='pdf', view=False)

# %%
# Sample from posterior.
with model:
    trace = pm.sample(1000, tune=4000, target_accept=0.9, random_seed=RANDOM_SEED)
    idata = az.from_pymc3(trace, model=model)

post = idata.posterior = idata.posterior.rename_vars(
    {
        "chol_user_corr": "Rho_user",
        "chol_user_stds": "sigma_user",
        "chol_block_corr": "Rho_block",
        "chol_block_stds": "sigma_block",
    }
)

# %%
# Save data about Box Cox transformation.
transform_report = reports_path / 'BoxCox.txt'
with transform_report.open(mode='w') as f:
    f.write(f"lambda={fitted_lambda}\n")
    f.write(f"min={boxcox_ci[0]}\n")
    f.write(f"max={boxcox_ci[1]}\n")
logging.info(f"Written Box Cox transform report to {transform_report.resolve()}")

# Save model and InferenceData to disk.
models_path = Path.cwd() / 'models'
with open(models_path / 'hlm2_model.pkl', 'wb') as buff:
    pickle.dump(model, buff)
idata.to_netcdf(models_path / 'hlm2_idata.nc')
