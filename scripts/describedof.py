# notebook: 02-oh-describe_dof.ipynb

# %% [markdown]
# # Descriptive Statistics of Degrees of Freedom
# 

# %%
# Basic imports and setup.

import sys
import logging
from pathlib import Path

import pandas as pd
import plotly.io as pio

from neuropsymodelcomparison.dataprocessing import analysis
from neuropsymodelcomparison import plot

# Default file format for figures.
pio.kaleido.scope.default_format = "pdf"

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# %% [markdown]
# Get preprocessed trials data.

# %%
data_path = Path.cwd() / 'data/preprocessed'
trials_filepath = data_path / 'trials.csv'
df = pd.read_csv(trials_filepath, index_col='id')
# Easier on memory and faster groupby.
df[['user', 'session', 'block', 'block_id', 'condition', 'task']] = df[['user', 'session', 'block', 'block_id', 
                                                                        'condition', 'task']].astype('category')

# When we view statistics by task, we want them to to be displyed in a certain order.
task_display_order = ['pre', 'df1', 'df2', 'df1|df2', 'post']

# %% [markdown]
# ## Analysis of Slider Usage
# %% [markdown]
# Onset of sliders for df1 and df2 being grabbed.

# %%
fig_onset = plot.generate_violin_figure(df[['user', 'condition', 'block', 'task', 'df1_grab', 'df2_grab']].rename(
                                        columns={'df1_grab': 'df1', 'df2_grab': 'df2'}),
                                        ['df1', 'df2'], ytitle="Grab Onset (s)", legend_title="DOF", width=1000)
        
# %%
onset_stats = df.groupby('task', sort=False)[['df1_grab', 'df2_grab']].describe().stack(level=0).T[task_display_order]

# %% [markdown]
# Duration of sliders for df1 and df2 being grabbed.

# %%
fig_duration = plot.generate_violin_figure(df[['user', 'condition', 'block', 'task', 'df1_duration', 'df2_duration']]\
                .rename(columns={'df1_duration': 'df1','df2_duration': 'df2'}),
                        ['df1', 'df2'], ytitle='Grab Duration (s)', legend_title="DOF", autosize=False, width=1000)

# %%
duration_stats = df.groupby('task', sort=False)[['df1_duration', 'df2_duration']]\
                    .describe().stack(level=0).T[task_display_order]

# %% [markdown]
# ## Normality Inspections
# ### QQ-Plots
# QQ-plots for comparing final states distribution of degrees of freedom to normal distributions.
# 
# The Filliben’s formula was used to estimate the theoretical quantiles for all QQ-plots.

# %%
fig_qq_dof = plot.generate_qq_plot(df, vars_=['df1', 'df2'], width=1000)
fig_qq_sum = plot.generate_qq_plot(df, vars_=['sum'], width=1000)

# %% [markdown]
# ### Histograms
# Histograms of final state values for df1 and df2 compared to normal distributions.

# %%
# Collect all histograms in a dictionary for later use of the keys as part of their file name when saving.
histograms = dict()

histograms['overall_dof'] = plot.generate_histograms(df[['df1', 'df2']], 
                                                     x_title="Final State Values",
                                                     legend_title="DOF",
                                                     width=1000)

# %%
for task, group in df.groupby('task', sort=False):
    histograms[task] = plot.generate_histograms(group[['df1', 'df2']], 
                                                x_title=f"Final State Values for Task \"{task}\"",
                                                legend_title="DOF",
                                                width=1000,
                                                xaxis_range=[0, 100])

# %%
final_state_stats = df.groupby('task')[['df1', 'df2']].describe().stack(level=0).T[task_display_order]

# %%
histograms['sum'] = plot.generate_histograms(df[['task', 'sum']], by='task', x_title="Final State Sum Values", legend_title="Block Type", width=1000)

# %%
sum_stats = df.groupby('task')['sum'].describe().T[task_display_order]

# %% [markdown]
# ## Save Reports

# %%
reports_path = Path.cwd() / 'reports'
# Save descriptive statistics for onset, duration, final states/sum.
# Flatten multiindex for saving.
onset_stats.index.rename('statistic', inplace=True)
onset_stats.columns = ["-".join(i) for i in onset_stats.columns.to_flat_index()]
out_file = reports_path / 'onset_stats.csv'
onset_stats.to_csv(out_file)
logging.info(f"Written report to {out_file.resolve()}")

duration_stats.index.rename('statistic', inplace=True)
duration_stats.columns = ["-".join(i) for i in duration_stats.columns.to_flat_index()]
out_file = reports_path / 'duration_stats.csv'
duration_stats.to_csv(out_file)
logging.info(f"Written report to {out_file.resolve()}")

final_state_stats = df.groupby('task')[['df1', 'df2', 'sum']].describe().stack(level=0).T[task_display_order]  
final_state_stats.index.rename('statistic', inplace=True)
final_state_stats.columns = ["-".join(i) for i in final_state_stats.columns.to_flat_index()]
out_file = reports_path / 'final_state_stats.csv'
final_state_stats.to_csv(out_file)
logging.info(f"Written report to {out_file.resolve()}")

# Save figures.
figures_path = reports_path / 'figures'
# Violin plots.
fig_filepath = figures_path / 'dof_onset.pdf'
fig_onset.write_image(str(fig_filepath))
logging.info(f"Written figure to {fig_filepath.resolve()}")

fig_filepath = figures_path / 'dof_duration.pdf'
fig_duration.write_image(str(fig_filepath))
logging.info(f"Written figure to {fig_filepath.resolve()}")

fig_filepath = figures_path / 'qq-plot_dof.pdf'
fig_qq_dof.write_image(str(fig_filepath))
logging.info(f"Written figure to {fig_filepath.resolve()}")

fig_filepath = figures_path / 'qq-plot_sum.pdf'
fig_qq_sum.write_image(str(fig_filepath))
logging.info(f"Written figure to {fig_filepath.resolve()}")

for hist_name, fig in histograms.items():
    fig_filepath = figures_path / f'histogram_{hist_name.replace("|", "-")}.pdf'
    fig.write_image(str(fig_filepath))
    logging.info(f"Written figure to {fig_filepath.resolve()}")


