# notebook source: 04-performance_variables.ipynb

# %% [markdown]
# # Analysis of Performance Variables
# 
# ## Task Performance I: Sum of Degrees of Freedom
# The first performance variable is the radius of the white circle, which is the sum of both degrees of freedom. The target is 125.

# %%
# Basic imports and setup.

import sys
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
import plotly.express as px
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
reports_path = Path.cwd() / 'reports'

trials_filepath = data_path / 'trials.csv'
df = pd.read_csv(trials_filepath, index_col='id')
# Easier on memory and faster groupby.
df[['user', 'session', 'block', 'block_id', 'condition', 'task']] = df[['user', 'session', 'block', 'block_id', 
                                                                        'condition', 'task']].astype('category')

# When we view statistics by task, we want them to to be displyed in a certain order.
task_display_order = ['pre', 'df1', 'df2', 'df1|df2', 'post']
df.task.cat.reorder_categories(task_display_order, inplace=True)
condition_display_order = ['df1', 'df2', 'df1|df2']
df.condition.cat.reorder_categories(condition_display_order, inplace=True)

# %% [markdown]
# ## Distribution
# ### Normality Inspections
# 
# QQ-plots for comparing final states distribution of degrees of freedom to normal distributions.
# 
# The Filliben’s formula was used to estimate the theoretical quantiles for all QQ-plots.

# %%
fig_qq_sum = plot.generate_qq_plot(df, vars_=['sum'], width=1000)

# %% [markdown]
# #### With Confidence Intervals

# %%

fig_qq_sum_ci, axes = plt.subplots(1, len(task_display_order), figsize=(14, 4))
for i, task in enumerate(task_display_order):
    pg.qqplot(df[df['task'] == task]['sum'], dist='norm', ax=axes[i])
    axes[i].set_title(f"Task={task}")
fig_qq_sum_ci.tight_layout()
plt.savefig(reports_path / 'figures/qq-plot-sum_ci.pdf')

# %% [markdown]
# ### Histogram

# %%
fig_hist_sum = plot.generate_histograms(df[['task', 'sum']], by='task', x_title="Final State Sum Values", 
                                        legend_title="Block Type", width=1000)

# %% [markdown]
# ### Statistics

# %%
sum_stats = df.groupby('task')['sum'].describe().T[task_display_order]

# %% [markdown]
# ### Visualisations of Means and Variability

# %%
# Use a barplot instead of a boxplot, because it more clearly conveys the core statistics we're interested in.
fig_sum_stats = px.bar(sum_stats.T.reset_index(), x='task', y='mean', error_y='std', 
                       labels={'mean': 'Sum Mean', 'task': 'Task'}, width=500)

# %%
sum_stats_by_condition = df.groupby(['condition', 'block'], sort=False, observed=True)['sum']\
                           .agg(['mean', 'std', 'count']).loc[condition_display_order]
fig_sum_by_condition = px.bar(sum_stats_by_condition.reset_index(), x='block', y='mean', error_y='std',
                              facet_col='condition', labels={'mean': 'Sum Mean', 'block': 'Block'},
                              category_orders={'condition': ['df1', 'df2', 'df1|df2']})

# %% [markdown]
# #### By Participant

# %%
df_block_stats = pd.read_csv(reports_path / 'block_stats.csv', index_col='block_id')

# %%
fig_sum_by_user = plot.generate_means_figure(df_block_stats, width=1000)

# %% [markdown]
# ## Task Performance II: Mean final state difference
# The difference in mean values of degrees of freedom is the performance variable for the additional task, 
# since the solution to the treatment tasks is that both degrees of freedom are at 62.5.
# 
# ###   Per degree of freedom and participant. Vertical bars represent standard deviations.

# %%
# ToDo: task 2 performance variable: difference of means

# %% [markdown]
# # Save Reports

# %%
out_file = reports_path / 'sum_stats_by_condition.csv'
sum_stats_by_condition.reset_index().to_csv(out_file, index=False)
logging.info(f"Written report to {out_file.resolve()}")

# Save figures.
figures_path = reports_path / 'figures'
# QQ-Plots
fig_filepath = figures_path / 'qq-plot-sum.pdf'
fig_qq_sum.write_image(str(fig_filepath))
logging.info(f"Written figure to {fig_filepath.resolve()}")
# Histogram

fig_filepath = figures_path / 'histogram-sum.pdf'
fig_hist_sum.write_image(str(fig_filepath))
logging.info(f"Written figure to {fig_filepath.resolve()}")

# Barplots
fig_filepath = figures_path / 'barplot-sum.pdf'
fig_sum_stats.write_image(str(fig_filepath))
logging.info(f"Written figure to {fig_filepath.resolve()}")

fig_filepath = figures_path / 'barplot-sum_by_condition.pdf'
fig_sum_by_condition.write_image(str(fig_filepath))
logging.info(f"Written figure to {fig_filepath.resolve()}")

fig_filepath = figures_path / 'barplot-sum_by_user.pdf'
fig_sum_by_user.write_image(str(fig_filepath))
logging.info(f"Written figure to {fig_filepath.resolve()}")
