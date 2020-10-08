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
import plotly.graph_objects as go
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
df = pd.read_csv(trials_filepath, index_col='id', dtype={'outlier': bool, 'exclude': bool})
# Clear outliers and excluded trials.
df = df.loc[~(df['outlier'] | df['exclude'])].drop(['outlier', 'exclude'], axis='columns')
# Easier on memory and faster groupby.
df[['user', 'session', 'block', 'block_id', 'condition', 'task']] = df[['user', 'session', 'block', 'block_id', 
                                                                        'condition', 'task']].astype('category')

# When we view statistics by task, we want them to to be displyed in a certain order.
task_display_order = ['pre', 'df1', 'df2', 'df1|df2', 'post']
df.task.cat.reorder_categories(task_display_order, inplace=True)
condition_display_order = ['df1', 'df2', 'df1|df2']
df.condition.cat.reorder_categories(condition_display_order, inplace=True)

# %%
df_block_stats = pd.read_csv(reports_path / 'block_stats.csv', index_col='block_id', dtype={'exclude': bool})
df_block_stats = df_block_stats.loc[~df_block_stats['exclude']].drop('exclude', axis='columns')
df_block_stats[['user', 'session', 'condition', 'block', 'task']] = df_block_stats[['user', 'session', 'condition', 
                                                                                    'block', 'task']].astype('category')
# %% [markdown]
# ## Difficulty Rating

# %%
fig_rating_by_block = px.line(df_block_stats, x='block', y='rating', line_group='user', line_dash='condition',
                              color='condition', category_orders={'condition': condition_display_order}, 
                              labels={'block': 'Block', 'rating': 'Difficulty Rating', 'condition': 'Condition'},
                              height=500)
fig_rating_by_block.update_yaxes(hoverformat='.2f',
                                 tickvals=df_block_stats['rating'].unique(),
                                 ticktext=['Very Easy', 'Easy', 'Neutral', 'Difficult', 'Very Difficult'])
fig_rating_by_block.update_xaxes(tickvals=df_block_stats['block'].unique(),
                                 range=[df_block_stats['block'].unique().as_ordered().min()-0.2,
                                        df_block_stats['block'].unique().as_ordered().max()+0.2])
legend = go.layout.Legend(xanchor='right',
                          yanchor='top',
                          orientation='v',
                          title="Condition")
fig_rating_by_block.update_layout(margin=plot.theme['graph_margins'], legend=legend)

# %% [markdown]
# ### Does difficulty rating correlate with performance?
# %%
fig_rating_vs_var = px.scatter(df_block_stats, x='rating', y='sum variance', color='condition', 
                               category_orders={'condition': condition_display_order},
                               hover_data=['user', 'block'])
legend = go.layout.Legend(xanchor='right',
                          yanchor='top',
                          orientation='v',
                          title="Condition")
fig_rating_vs_var.update_yaxes(hoverformat='.2f')
fig_rating_vs_var.update_xaxes(tickvals=df_block_stats['rating'].unique(), 
                               ticktext=['Very Easy', 'Easy', 'Neutral', 'Difficult', 'Very Difficult'])
fig_rating_vs_var.update_layout(margin=plot.theme['graph_margins'], legend=legend)

# %%
sumvar_rating_levene = pg.homoscedasticity(df_block_stats, dv='sum variance', group='rating')

# %% [markdown]
# ## Distribution
# Let's first look at the mean final states in relation to the task goals.

# %%
fig_mean_scatter = px.scatter(df_block_stats, x='df1 mean', y='df2 mean', symbol='block', color='task',
                              category_orders={'task': task_display_order},
                              hover_data={'user': True, 'sum variance': ':.2f', 'sum mean': ':.2f', 'rating': ':d'})
# Task goal 1 visualization.
fig_mean_scatter.add_scatter(
    x=[25, 100],
    y=[100, 25],
    mode='lines',
    name="task goal 1",
    marker={'color': 'black',
            },
    opacity=0.5,
    hovertemplate="df1+df2=125",
)
# Task goal 2 (DoF constrained) visualization.
fig_mean_scatter.add_scatter(y=[62.5], x=[62.5],
                name="task goal 2",
                hovertemplate="df1=df2=62.5",
                mode='markers',
                marker_symbol='x',
                opacity=0.5,
                marker={'size': 15,
                        'color': 'black'})
fig_mean_scatter.update_xaxes(hoverformat='.2f', title='Degree of Freedom 1', range=[0, 100], constrain='domain')
fig_mean_scatter.update_yaxes(hoverformat='.2f', title='Degree of Freedom 2', range=[0, 100],
                              scaleanchor='x', scaleratio=1)
legend = go.layout.Legend(
        xanchor='right',
        yanchor='top',
        orientation='v',
        title="Task, Block"
    )
fig_mean_scatter.update_layout(margin=plot.theme['graph_margins'], legend=legend, width=540)

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
fig_sum_stats.add_trace(go.Scatter(x=[task_display_order[0], task_display_order[-1]],
                                   y=[125, 125],
                                   name=f"Target",
                                   showlegend=True,
                                   hovertext=f'Target',
                                   hoverinfo='y',
                                   textposition='top center',
                                   mode='lines',
                                   marker={'color': 'red'}
                                   )
                        )

# %%
sum_stats_by_condition = df.groupby(['condition', 'block'], sort=False, observed=True)['sum']\
                           .agg(['mean', 'std', 'count']).loc[condition_display_order]
fig_sum_by_condition = px.bar(sum_stats_by_condition.reset_index(), x='block', y='mean', error_y='std',
                              facet_col='condition', labels={'mean': 'Sum Mean', 'block': 'Block'},
                              category_orders={'condition': ['df1', 'df2', 'df1|df2']})
for col in range(1, sum_stats_by_condition.index.unique(level='condition').size + 1):
    fig_sum_by_condition.add_trace(go.Scatter(x=[0.5, 3.5],
                                              y=[125, 125],
                                              name=f"Target",
                                              showlegend=col==1,
                                              hovertext=f'Target',
                                              hoverinfo='y',
                                              textposition='top center',
                                              mode='lines',
                                              marker={'color': 'red'}
                                              ), row=1, col=col
                                  )

# %% [markdown]
# #### By Participant

# %%
fig_sum_by_user = plot.generate_means_figure(df_block_stats, width=1000)

# %% [markdown]
# ## Task Performance II: Mean final state difference
# The difference in mean values of degrees of freedom is the performance variable for the additional task, 
# since the solution to the treatment tasks is that both degrees of freedom are at 62.5.
# 
# ###   Difference from target value per degree of freedom

# %%
target_diff = df[['user', 'block', 'condition', 'task', 'df1', 'df2']]
target_diff[['df1', 'df2']] = target_diff[['df1', 'df2']] - 62.5

fig_target_diff_hist = plot.generate_histogram_rows(target_diff.loc[df['block']==2].drop(['user', 'block', 'task'],
                                                                                         axis='columns'),
                                                    x_title="Difference to Target", 
                                                    rows='condition',
                                                    legend_title="Variable")

# %%
fig_target_diff_violin = plot.generate_violin_figure(target_diff, ['df1', 'df2'],
                                                     "Difference to Target",
                                                     legend_title="DoF",
                                                     width=1000)

# %% [markdown]
# ### Difference between degrees of freedom.
# The intend to treat is that both degrees of freedom are equal.
# This may be achieved without reaching the intended target value precisely.

# %%
df['diff'] = df[['df1', 'df2']].diff(axis='columns').dropna(axis='columns').abs().squeeze()
dof_diff = df.groupby(['block', 'condition'], observed=True)['diff'].describe()

# %%
fig_df_diff = px.box(df, x='block', y='diff', facet_col='condition', notched=True, color='block',
                     category_orders={'condition':condition_display_order},
                     labels={'diff': "DoF Difference (absolute)", 'block': 'Block'},
                     hover_data=['user'])
fig_df_diff.update_yaxes(hoverformat='.2f')
legend = go.layout.Legend(xanchor='right',
                          yanchor='top',
                          orientation='v',
                          title="Block")
fig_df_diff.update_layout(margin=plot.theme['graph_margins'], legend=legend)

# %% [markdown]
# #### Equivalence test
# If we take a 5% error margin for the differece between degrees of freedom, 
# that translates to a tolerance of 5 units between degrees of freedom.
dof_tost = df.groupby(['condition', 'block']).apply(lambda x: pg.tost(x=x['df1'], y=x['df2'], bound=5, paired=True))

# %% [markdown]
# # Save Reports

# %%
def save_report(dataframe, filename, index=True):
    """ Save dataframe as CSV to reports. """
    out_file = reports_path / filename
    dataframe.to_csv(out_file, index=index)
    logging.info(f"Written report to {out_file.resolve()}")

save_report(sum_stats_by_condition.reset_index(), 'sum_stats_by_condition.csv', index=False)
save_report(sumvar_rating_levene, 'sumvar_rating_levene.csv')
save_report(dof_diff.reset_index(), 'dof_difference.csv', index=False)
save_report(dof_tost.reset_index(), 'dof_tost.csv', index=False)

target_diff_stats = target_diff.groupby(['condition', 'block'], observed=True).agg(['mean', 'std'])
target_diff_stats.columns = [' '.join(c).strip() for c in target_diff_stats.columns]
target_diff_stats.reset_index(inplace=True)
save_report(target_diff_stats, 'dof_diff_goal2.csv', index=False)

# Save figures.
figures_path = reports_path / 'figures'

def save_fig(fig, filename):
    """ Save plotly figure to reports. """
    fig_filepath = figures_path / filename
    fig.write_image(str(fig_filepath))
    logging.info(f"Written figure to {fig_filepath.resolve()}")

# Scatter
save_fig(fig_rating_by_block, 'line-plot-rating_block.pdf')
save_fig(fig_rating_vs_var, 'scatter-rating_variance.pdf')
save_fig(fig_mean_scatter, 'scatter-sum_mean.pdf')
# QQ-Plots
save_fig(fig_qq_sum, 'qq-plot-sum.pdf')
# Histogram
save_fig(fig_hist_sum, 'histogram-sum.pdf')
save_fig(fig_target_diff_hist, 'histogram-sum.pdf')
# Violin
save_fig(fig_target_diff_violin, 'violin-target_diff.pdf')
# Barplots
save_fig(fig_sum_stats, 'barplot-sum.pdf')
save_fig(fig_sum_by_condition, 'barplot-sum_by_condition.pdf')
save_fig(fig_sum_by_user, 'barplot-sum_by_user.pdf')
save_fig(fig_df_diff, 'boxplot-dof_diff.pdf')
