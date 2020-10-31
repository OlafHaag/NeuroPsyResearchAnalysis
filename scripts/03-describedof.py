# notebook source: 03-oh-describe_dof.ipynb

# %% [markdown]
# # Descriptive Statistics of Degrees of Freedom
# 

# %%
# Basic imports and setup.

import sys
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import plotly.express as px
import plotly.io as pio
import seaborn as sns

from neuropsymodelcomparison.dataprocessing import analysis
from neuropsymodelcomparison import plot

# Default file format for figures.
pio.kaleido.scope.default_format = "pdf"
pio.templates.default = "plotly_white"

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# %% [markdown]
# Get preprocessed trials data.

# %%
data_path = Path.cwd() / 'data/preprocessed'
reports_path = Path.cwd() / 'reports'

trials_filepath = data_path / 'trials.csv'
df = pd.read_csv(trials_filepath, index_col='id', dtype={'outlier': bool, 'exclude': bool})
# Clear outliers and excluded trials.
df = df.loc[~(df['outlier'] | df['exclude'])].drop(['outlier', 'exclude'], axis='columns')
# Easier on memory and faster groupby.
df[['user', 'session', 'block', 'block_id', 'condition', 'task']] = df[['user', 'session', 'block', 'block_id', 
                                                                        'condition', 'task']].astype('category')

# When we view statistics by task, we want them to to be displayed in a certain order.
task_display_order = ['pre', 'df1', 'df2', 'df1|df2', 'post']
df.task.cat.reorder_categories(task_display_order, inplace=True)
condition_display_order = ['df1', 'df2', 'df1|df2']
df.condition.cat.reorder_categories(condition_display_order, inplace=True)

# %% [markdown]
# ## Analysis of Slider Usage
# %% [markdown]
# Onset of sliders for df1 and df2 being grabbed.

# %%
fig_onset = plot.generate_violin_figure(df[['user', 'condition', 'block', 'task', 'df1_grab', 'df2_grab']]\
                                        .rename(columns={'df1_grab': 'df1', 'df2_grab': 'df2'}), 
                                        ['df1', 'df2'], ytitle="Grab Onset (s)", legend_title="DOF", 
                                        width=600, height=300)
        

# %%
onset_stats = df.groupby('task', sort=False)[['df1_grab', 'df2_grab']].describe().stack(level=0).T[task_display_order]

# %% [markdown]
# Duration of sliders for df1 and df2 being grabbed.

# %%
fig_duration = plot.generate_violin_figure(df[['user', 'condition', 'block', 'task', 'df1_duration', 'df2_duration']]\
                                           .rename(columns={'df1_duration': 'df1','df2_duration': 'df2'}),
                                           ['df1', 'df2'], ytitle='Grab Duration (s)', legend_title="DOF",
                                           autosize=False, width=600, height=300)

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
fig_qq_dof = plot.generate_qq_plot(df, vars_=['df1', 'df2'], width=600, height=300)

# %% [markdown]
# #### Separately

# %%

fig_qq_grid, axes = plt.subplots(2, len(task_display_order), figsize=(14, 8))
for i, task in enumerate(task_display_order):
    pg.qqplot(df[df['task'] == task]['df1'], dist='norm', ax=axes[0, i])
    pg.qqplot(df[df['task'] == task]['df2'], dist='norm', ax=axes[1, i])
    axes[0, i].set(ylabel='Ordered quantiles (df1)', title=f"Task={task}")
    axes[1, i].set(ylabel='Ordered quantiles (df2)', title="")
fig_qq_grid.tight_layout()
plt.savefig(reports_path / 'figures/qq-plot-dof-grid.pdf')

# %% [markdown]
# ### Histograms
# Histograms of final state values for df1 and df2 compared to normal distributions.

# %%
# Collect all histograms in a dictionary for later use of the keys as part of their file name when saving.
histograms = dict()

histograms['overall_dof'] = plot.generate_histograms(df[['df1', 'df2']], x_title="Final State Values",
                                                     legend_title="DOF", width=600, height=300)

# %%
for task, group in df.groupby('task', sort=False):
    histograms[task] = plot.generate_histograms(group[['df1', 'df2']],
                                                x_title=f"Final State Values for Task \"{task}\"", legend_title="DOF",
                                                width=600, height=300, xaxis_range=[0, 100])

# %%
final_state_stats = df.groupby('task')[['df1', 'df2']].describe().stack(level=0).T[task_display_order]

# %% [markdown]
# The solution to the treatment tasks is that both degrees of freedom are at 62.5.
# 
# ###   Per degree of freedom and participant. Vertical bars represent standard deviations.

# %%
df_block_stats = pd.read_csv(reports_path / 'block_stats.csv', index_col='block_id', dtype={'exclude': bool})
df_block_stats = df_block_stats.loc[~df_block_stats['exclude']].drop('exclude', axis='columns')

# %%
# Add standard deviation for use in plots.
df_block_stats[['df1 std', 'df2 std']] = df_block_stats[['df1 variance', 'df2 variance']].transform(np.sqrt)
# Convert to long format for easier plotting.
df_block_stats_long = analysis.wide_to_long(df_block_stats, stubs=['df1', 'df2'], suffixes=['mean', 'std'], j='dof')
fig_dof_line = plot.generate_lines_plot(df_block_stats_long, "mean", by='user', color_col='dof', errors='std',
                                        jitter=True, category_orders={'condition': condition_display_order},
                                        width=600, height=300)

# %% [markdown]
# ### Across Participants

# %%
fig_dof_line2 = sns.catplot(x="Block", y="Mean", hue="DoF", col="condition", col_order=condition_display_order, 
                            data=df_block_stats_long.rename(columns={'mean': 'Mean', 'block': 'Block', 'dof': 'DoF'}),
                            kind="point", dodge=True, width=10, aspect=.7,
                            palette={'df1': 'cornflowerblue','df2': 'palevioletred'})
figures_path = reports_path / 'figures'
fig_filepath = figures_path / 'line-plot-dof_mean.pdf'
plt.savefig(str(fig_filepath))
logging.info(f"Written figure to {fig_filepath.resolve()}")

# %%
fig_dof_violin = plot.generate_violin_figure(df_block_stats.rename(columns={'df1 mean': 'df1', 'df2 mean': 'df2'}), 
                                             columns=['df1', 'df2'], ytitle='Mean', legend_title="DOF",
                                              width=800)

# %% [markdown]
# ## Save Reports

# %%
# Save descriptive statistics for onset, duration, final states.
# LaTeX
out_file = reports_path / 'onset_table.tex'
onset_tab = onset_stats.T.rename_axis(["Task", "DoF"]).reset_index()
onset_tab['DoF'] = onset_tab['DoF'].str.split('_').apply(pd.Series)[0]
onset_tab.to_latex(out_file, caption="Onset (s) of Slider Use", label="tab:onset", float_format="%.2f", index=False)
logging.info(f"Written report to {out_file.resolve()}")
# Flatten multiindex for saving to CSV.
onset_stats.index.rename('statistic', inplace=True)
onset_stats.columns = ["-".join(i) for i in onset_stats.columns.to_flat_index()]
out_file = reports_path / 'onset_stats.csv'
onset_stats.to_csv(out_file)
logging.info(f"Written report to {out_file.resolve()}")

# LaTeX
out_file = reports_path / 'duration_table.tex'
duration_tab = duration_stats.T.rename_axis(["Task", "DoF"]).reset_index()
duration_tab['DoF'] = duration_tab['DoF'].str.split('_').apply(pd.Series)[0]
duration_tab.to_latex(out_file, caption="Duration (s) of Slider Use", label="tab:duration", float_format="%.2f", index=False)
logging.info(f"Written report to {out_file.resolve()}")
# CSV
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
# Violin plots.
fig_filepath = figures_path / 'violin-dof_onset.pdf'
fig_onset.write_image(str(fig_filepath))
logging.info(f"Written figure to {fig_filepath.resolve()}")

fig_filepath = figures_path / 'violin-dof_duration.pdf'
fig_duration.write_image(str(fig_filepath))
logging.info(f"Written figure to {fig_filepath.resolve()}")

fig_filepath = figures_path / 'violin-dof_mean.pdf'
fig_dof_violin.write_image(str(fig_filepath))
logging.info(f"Written figure to {fig_filepath.resolve()}")
# QQ-Plots
fig_filepath = figures_path / 'qq-plot-dof.pdf'
fig_qq_dof.write_image(str(fig_filepath))
logging.info(f"Written figure to {fig_filepath.resolve()}")
# Line-Plots
fig_filepath = figures_path / 'line-plot-dof_mean_by_user.pdf'
fig_dof_line.write_image(str(fig_filepath))
logging.info(f"Written figure to {fig_filepath.resolve()}")

# Histograms
for hist_name, fig in histograms.items():
    fig_filepath = figures_path / f'histogram-{hist_name.replace("|", "-")}.pdf'
    fig.write_image(str(fig_filepath))
    logging.info(f"Written figure to {fig_filepath.resolve()}")


