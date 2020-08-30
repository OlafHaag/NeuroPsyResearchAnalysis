# notebook source: 07-oh-modelcomparison.ipynb
# Todo: Explain what's going on here.

# %% [markdown]
# # Bayesian Model Comparison

# %%
# Basic imports and setup.

from configparser import ConfigParser
import sys
import logging
from pathlib import Path

import arviz
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.io as pio

from neuropsymodelcomparison.dataprocessing.modelcomparator import ModelComparison

# Default file format for plotly figures.
pio.kaleido.scope.default_format = "pdf"

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# %% [markdown]
# ## Prepare Data

# %%
# Common folder paths.
data_path = Path.cwd() / 'data/preprocessed'
reports_path = Path.cwd() / 'reports'
figures_path = reports_path / 'figures'

# Read in data.
# We want to use the same threshold for including sessions as we used previously. So all should be included.
try:
    sample_info = Path.read_text(reports_path / 'sampling.txt')
    sample_info = '[dummy_section]\n'+ sample_info
except IOError:
    min_trials = 18  # Fallback: 60% of 30 trials.
else:
    config_parser = ConfigParser()
    config_parser.read_string(sample_info)
    min_trials = int(config_parser.get('dummy_section', 'trials_count_threshold', fallback=18))

df = pd.read_csv(data_path / 'trials.csv', index_col='id')
# We only analyze the first session of each participant.
df = df.loc[df['session'] == 1, ['user', 'block', 'parallel', 'orthogonal']]


# %%
model_comp = ModelComparison(df, min_samples=min_trials)

# %% [markdown]
# ## Compute Posteriors

# %%
# Compare the data of each user to our theoretical models.
# This will take quite some time!
for user in df['user'].unique():
    model_comp.compare_models(user)

# %% [markdown]
# ## Visualize Results

# %%
fig_posteriors = px.imshow(model_comp.posteriors, labels=dict(x="Model", y="Participant", 
                           color="Posterior<br>Probability"), color_continuous_scale='Greys',
                           zmin=0, zmax=1, aspect='equal')
fig_posteriors.update_xaxes(side="top")
fig_posteriors.update_yaxes(tickmode='array', tickvals=model_comp.posteriors.index)

# %% [markdown]
# ## Save Reports

# %%
# Save tables.
output_file = reports_path / "posteriors.csv"
model_comp.write_posteriors(output_file)
logging.info(f"Written report to {output_file.resolve()}")

# Save Figures
fig_filepath = figures_path / 'heatmap-posteriors.pdf'
fig_posteriors.write_image(str(fig_filepath))
logging.info(f"Written figure to {fig_filepath.resolve()}")
