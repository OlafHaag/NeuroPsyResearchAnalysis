# notebook source: 02-oh-analyze_sample.ipynb

# %% [markdown]
# # Analyze Sample
# We'll detect outliers and remove them. Participants who didn't perform the tasks sufficiently well will be excluded 
# from further analyses. It's important to note, however, that for these individuals the task was either to difficult 
# to perform, or the instructions didn't convey the proper execution of the task for these individuals.

# %%
# Basic imports and setup.

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import shutil

from neuropsymodelcomparison.dataprocessing import analysis
from neuropsymodelcomparison import plot

# Default file format for figures.
pio.kaleido.scope.default_format = "pdf"
pio.templates.default = "plotly_white"

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# %% [markdown]
# Get preprocessed trials data.

# %%
data_path = Path.cwd() / 'data'
interim_path = data_path / 'interim'
trials_filepath = interim_path / 'trials.csv'
df = pd.read_csv(trials_filepath, index_col='id')
# Easier on memory and faster groupby.
df[['user', 'session', 'block', 'block_id', 'condition', 'task']] = df[['user', 'session', 'block', 'block_id', 
                                                                        'condition', 'task']].astype('category')

# When we view statistics by task, we want them to to be displayed in a certain order.
task_display_order = ['pre', 'df1', 'df2', 'df1|df2', 'post']
df.task.cat.reorder_categories(task_display_order, inplace=True)

# %% [markdown]
# ## Outlier Detection
# Outlier detection by covariance estimation in a Gaussian distributed dataset.

# %%
# Detect outliers in whole dataset.
contamination = 0.056  # Proportion of outliers in the data set.

# ToDo: Maybe detect outliers per session since we do the analyses mainly per session?
outliers, z = analysis.get_outlyingness(df[['df1', 'df2']].values, contamination=contamination)
df['outlier'] = outliers.astype(bool)
n_trials_outliers = df['outlier'].value_counts()[True]
# Todo: Exclude if mean sum is not in the vicinity 100-150 after outlier removal?

# %% [markdown]
# ### Plot Final States with Outlier Threshold
# 
# Display point clouds of data for final state values of degrees of freedom 1 and 2, colored by block. 
# The subspace of task goal 1 is presented as a line. The goal for the additional task is represented as a red cross.

# %%
fig_trials_scatter = plot.generate_trials_figure(df, contour_data=z)

# %% [markdown]
# ## Exclude sessions with insufficient data
# If a user didn't perform at least 60% of trials in a session sufficiently well, 
# exclude that user's session from further analyses.

# %%
# Minimum number of trials per block for being included in the analyses.
trials_proportion_theshold = 0.6  # proportion of total trials per block.
trials_total_count = 30
trials_count_threshold = int(trials_total_count * trials_proportion_theshold)

# Keep track of how many sessions there are before filtering.
n_sessions = df.groupby(['user', 'session'], observed=True).ngroups

# %%
# Aggregate valid trial counts.
df_counts = df.loc[~df['outlier']].groupby(['user', 'session', 'block_id', 'block', 'condition'], observed=True)\
              .size().rename('valid trials count').reset_index()

# Display some more information about users.
users = pd.read_csv(data_path / 'raw/users.csv')  # When using Int8DType for gaming_exp NAType causes TypeError in plot.
df_counts['gender'] = df_counts['user'].map(users['gender'])
df_counts['age_group'] = df_counts['user'].map(users['age_group'])
df_counts['gaming_exp'] = df_counts['user'].map(users['gaming_exp'])
# How did they rate the block? Can use Int8Dtype since answer is mandatory and hence no NAType is present.
blocks = pd.read_csv(data_path / 'raw/blocks.csv', index_col='id', dtype={'rating': pd.Int8Dtype()})
df_counts['rating'] = df_counts['block_id'].map(blocks['rating'])


# %%
# Bar plot.
fig_exclusions = px.bar(df_counts, x='user', y='valid trials count', color='block', barmode='group', opacity=0.9, 
                        hover_data=['condition', 'gender', 'age_group', 'gaming_exp', 'rating'],
                        labels=dict(zip(df_counts.columns, df_counts.columns.str.title())),
                        width=800)
fig_exclusions.update_layout(bargap=0.3, bargroupgap=0.01)

# Add threshold.
fig_exclusions.add_trace(go.Scatter(
    x=df_counts['user'],
    y=[trials_count_threshold] * len(df_counts),
    mode='lines',
    name="Inclusion Threshold",
    marker={'color': 'black'},
    opacity=0.8,
    hovertemplate=f"minimum count: {trials_count_threshold}",
))

# Mark excluded.
fig_exclusions.add_trace(go.Scatter(
    x=df_counts.loc[df_counts['valid trials count'] < trials_count_threshold, :]['user'],
    y=[trials_count_threshold] * len(df_counts.loc[df_counts['valid trials count'] < trials_count_threshold, :]),
    mode='markers',
    marker_symbol='x',
    marker_size=15,
    name="Excluded",
    marker={'color': 'black'},
    hovertemplate="Excluded",
))
# Make legend horizontal.
fig_exclusions.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))


# %%
# Mark sessions with insufficient data from dataframe.
df['exclude'] = (df.set_index(['user', 'session'])
                   .index.map(~df.loc[~df['outlier']].groupby(['user', 'session'], observed=True)
                                           .apply(lambda x: (x['block'].value_counts() >= trials_count_threshold).all())
                             )
                )
# For some unknown reason we have to assign bool dtype after creating the column.
# Doing it on the left side of above assignment didn't stick.
df['exclude'] = df['exclude'].astype(bool)
n_sessions_excluded_by_trials_count = n_sessions - df.loc[~df['exclude']].groupby(['user', 'session']).ngroups

# %% [markdown]
# ## Describe Remaining Sample
# Show demographic characteristics of remaining sample.

# %%
users = users.loc[users.index.isin(df.loc[~df['exclude'], 'user'].unique()), :]
print("Gender Distribution:\n")
print(users['gender'].value_counts(dropna=False), "\n")
print("Age Distribution:\n")
print(users['age_group'].value_counts(dropna=False))

# %% [markdown]
# ## Calculate Projection Lengths
# We're ultimately interested in the variance of final states in the directions parallel and orthogonal 
# to the uncontrolled manifold. 

# %%
# Calculate projections onto vectors parallel and orthogonal to UCM.
ucm_vec = analysis.get_ucm_vec()
projections = df.loc[~df['outlier']].groupby(['user', 'session', 'task'], observed=True)[['df1', 'df2']]\
                .apply(analysis.get_projections, ucm_vec)
df = pd.concat([df, projections], axis='columns')

# %% [markdown]
# ## Compute Statistics and Synergy Indices
# per participant

# %%
df_stats = analysis.get_statistics(df.dropna())
# Flatten hierarchical columns.
df_stats.columns = [" ".join(col).strip() for col in df_stats.columns.to_flat_index()]
df_stats.reset_index(inplace=True)
# Add users' rating of block's difficulty.
df_stats['rating'] = df_stats['block_id'].map(df_counts.set_index('block_id')['rating']).astype(int)
df_stats['exclude'] = df.drop_duplicates(subset=['user', 'session', 'block'])['exclude'].values

# %% [markdown]
# ## Save Data

# %%
# Save final state data with projections.
destination_path = data_path / 'preprocessed'
df.to_csv(destination_path / 'trials.csv')
logging.info(f"Written preprocessed trials data to {(destination_path / 'trials.csv').resolve()}")

# Save final state position and projection in long format.
df_long = df.melt(id_vars=['user', 'session', 'condition', 'block_id', 'block', 'trial', 'exclude'],
                  value_vars=['df1', 'df2'],
                  var_name='dof', value_name='value')
df_long.index.rename('id', inplace=True)
df_long.to_csv(destination_path / 'dof_long.csv')
logging.info("Written preprocessed degrees of freedom final states in long format to "
             f"{(destination_path / 'dof_long.csv').resolve()}")

projections_long = df.loc[~df['outlier']].melt(id_vars=['user', 'session', 'condition',
                                                        'block_id', 'block', 'trial', 'exclude'],
                                               value_vars=['parallel', 'orthogonal'],
                                               var_name='direction', value_name='projection')
projections_long.index.rename('id', inplace=True)
projections_long.to_csv(destination_path / 'projections.csv')
logging.info(f"Written projection data in long format to {(destination_path / 'projections.csv').resolve()}")

reports_path = Path.cwd() / 'reports'
stats_path = reports_path / 'block_stats.csv'
df_stats.to_csv(stats_path, index=False)
logging.info(f"Written statistics about each block to {stats_path.resolve()}")

# Save data about exclusions. Make a copy and add content, so we can run this notebook separately from the previous one.
sampling_report = reports_path / 'interim/sampling.txt'
destination_path = sampling_report.parent.parent / sampling_report.name
shutil.copy(str(sampling_report), str(destination_path))  # With Python version < 3.9 we need to provide strings.

with destination_path.open(mode='a') as f:
    f.write(f"contamination={contamination}\n")
    f.write(f"trials_proportion_theshold={trials_proportion_theshold}\n")
    f.write(f"trials_count_threshold={trials_count_threshold}\n")
    f.write(f"n_trials_outliers={n_trials_outliers}\n")
    f.write(f"n_sessions_excluded_by_trials_count={n_sessions_excluded_by_trials_count}\n")
    f.write(f"final_n_users={len(users)}\n")
    f.writelines([f"final_gender_{k}={v}\n" for k,v in users['gender'].value_counts(dropna=False).iteritems()])
    f.writelines([f"final_age_{k}={v}\n" for k,v in users['age_group'].value_counts(dropna=False).iteritems()])

logging.info(f"Written sampling data to {destination_path.resolve()}")

# Save figures.
figures_path = reports_path / 'figures'
fig_trials_scatter_filepath = figures_path / 'scatter-outliers.pdf'
fig_trials_scatter.write_image(str(fig_trials_scatter_filepath))
logging.info(f"Written figure to {fig_trials_scatter_filepath.resolve()}")

fig_exclusions_filepath = figures_path / 'barplot-exclusions.pdf'
fig_exclusions.write_image(str(fig_exclusions_filepath))
logging.info(f"Written figure to {fig_exclusions_filepath.resolve()}")
