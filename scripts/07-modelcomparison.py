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
import plotly.graph_objects as go
import plotly.io as pio

from neuropsymodelcomparison.dataprocessing.modelcomparator import ModelComparison
from neuropsymodelcomparison import plot

# Default file format for plotly figures.
pio.kaleido.scope.default_format = "pdf"

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# %% [markdown]
# ## Prepare Data

# %%
# Common folder paths.
data_path = Path.cwd() / 'data'
reports_path = Path.cwd() / 'reports'
figures_path = reports_path / 'figures'

# Read in data.
# We want to use the same threshold for including sessions as we used previously. So all should be included.
# Alternatively, we could exclude by the 'exclude' column in the data.
try:
    sample_info = Path.read_text(reports_path / 'sampling.txt')
    sample_info = '[dummy_section]\n'+ sample_info
except IOError:
    min_trials = 18  # Fallback: 60% of 30 trials.
else:
    config_parser = ConfigParser()
    config_parser.read_string(sample_info)
    min_trials = int(config_parser.get('dummy_section', 'trials_count_threshold', fallback=18))

trial_data = pd.read_csv(data_path / 'preprocessed/trials.csv', index_col='id')
# We only analyze the first session of each participant.
df = trial_data.loc[trial_data['session'] == 1 & ~trial_data['outlier'], ['user', 'block', 'parallel', 'orthogonal']]


# %%
model_comp = ModelComparison(df, min_samples=min_trials)

# %% [markdown]
# ## Compute Posteriors

# %%
# Compare the data of each user to our theoretical models.
# This will take quite some time!
for user in model_comp.df['user'].unique():
    model_comp.compare_models(user)

# %%
# Augment posterior data.
columns = model_comp.posteriors.columns
# Condition.
conditions = trial_data.loc[trial_data['user'].isin(model_comp.posteriors.index), 
                            ['user','condition']].drop_duplicates().set_index('user')
model_comp.posteriors = model_comp.posteriors.join(conditions)
# Gaming experience.
users_path = data_path / 'raw/users.csv'
exp = pd.read_csv(users_path, dtype={'gaming_exp': pd.Int8Dtype()}).loc[model_comp.posteriors.index, 'gaming_exp']
model_comp.posteriors = model_comp.posteriors.join(exp)

# %% [markdown]
# ## Visualize Results

# %%
fig_posteriors = px.imshow(model_comp.posteriors.drop(['condition', 'gaming_exp'], 
                                                      axis='columns').reset_index(drop=True),
                           labels=dict(x="Model", y="Participant", color="Posterior<br>Probability"), 
                           color_continuous_scale='Greys', zmin=0, zmax=1,
                           aspect='equal', height=len(model_comp.posteriors)*30, width=500)
fig_posteriors.update_xaxes(side="top", showspikes=True, spikemode="across")
fig_posteriors.update_yaxes(tickmode='array', 
                            tickvals=list(range(len(model_comp.posteriors))),
                            ticktext=model_comp.posteriors.index,
                            showspikes=True)

# %%
# By condition.
post = model_comp.posteriors.melt(id_vars=['user', 'condition'],
                                  value_vars=columns.drop('user'),
                                  value_name='probability',
                                  var_name='model')

fig_post_hist =px.histogram(post, x='probability',
                            color='Model',
                            opacity=0.7,
                            facet_col='condition',
                            facet_row='Model',
                            histnorm='percent',
                            labels={'model': "Model",
                                    'probability': "Probability",
                                    'condition': "Condition"},
                            height=800)
fig_post_hist.update_yaxes(hoverformat='.2f', title="")
fig_post_hist.update_layout(showlegend=False,
                            margin=plot.theme['graph_margins'],
                            annotations = list(fig_post_hist.layout.annotations) + 
                            [go.layout.Annotation(x=-0.09,
                                                  y=0.5,
                                                  font=dict(
                                                      size=14
                                                  ),
                                                  showarrow=False,
                                                  text="Frequency (percent)",
                                                  textangle=-90,
                                                  xref="paper",
                                                  yref="paper"
                                                 )
                            ])

# %% [markdown]
# ## Save Reports

# %%
# Save tables.
out_file = reports_path / "posteriors.csv"
model_comp.write_posteriors(out_file)
logging.info(f"Written report to {out_file.resolve()}")

# Save figures.
fig_filepath = figures_path / 'heatmap-posteriors.pdf'
fig_posteriors.write_image(str(fig_filepath))
logging.info(f"Written figure to {fig_filepath.resolve()}")

fig_filepath = figures_path / 'histogram-posteriors.pdf'
fig_post_hist.write_image(str(fig_filepath))
logging.info(f"Written figure to {fig_filepath.resolve()}")

# Save traces.
out_path = Path.cwd() / 'models'
model_comp.write_traces(out_path)
logging.info(f"Written traces to {out_path.resolve()}")
