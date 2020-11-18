# notebook: 03-oh-pca-analysis.ipynb

# %% [markdown]
# # Principal Component Analysis
# We want to know if the principal components coincide with the uncontrolled manifold direction.

# %%
# Basic imports and setup.

import sys
import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from neuropsymodelcomparison.dataprocessing import analysis
from neuropsymodelcomparison import plot

# Default file format for figures.
pio.kaleido.scope.default_format = "pdf"
pio.templates.default = "plotly_white"

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# %% [markdown]
# Get preprocessed trials data.

# %%
data_path = Path.cwd() / 'data/preprocessed/trials.csv'
df = pd.read_csv(data_path, index_col='id', dtype={'outlier': bool, 'exclude': bool})
# Clear outliers and excluded trials.
df = df.loc[~(df['outlier'] | df['exclude'])].drop(['outlier', 'exclude'], axis='columns')
# Easier on memory and faster groupby.
df[['user', 'session', 'block', 'block_id', 'condition', 'task']] = df[['user', 'session', 'block', 'block_id',
                                                                        'condition', 'task']].astype('category')

# When we view statistics by task, we want them to to be displyed in a certain order.
task_display_order = ['pre', 'df1', 'df2', 'df1|df2', 'post']
df.task.cat.reorder_categories(task_display_order, inplace=True)

# %% [markdown]
# ## PCA per participant
# If each participant favors a different part of the solution space, we have to perform PCA for each participant 
# separately. Otherwise we risk only measuring variance across participants, but not within participants.
# %%
pca_results = df.groupby(['user', 'task']).apply(analysis.get_pca_data).sort_index(level='task')
pca_results.reset_index(inplace=True)
pca_results[['task', 'PC']] = pca_results[['task', 'PC']].astype('category')

# %%
# Get the summary.
var_expl_summary = pca_results.groupby(['task', 'PC'])['var_expl_ratio'].agg(['mean', 'std'])

# %%
pca_barplot = plot.generate_pca_figure(var_expl_summary.reset_index(), value='mean', error='std')

# %% [markdown]
# ## Differences between Principal Components and Uncontrolled Manifold
# We measure the differences between the directions of the principal components and the vectors parallel and orthogonal 
# to the UCM in degrees.

# %%
ucm_vec = analysis.get_ucm_vec()
angle_df = pca_results.groupby(['user', 'task'], observed=True, sort=False).apply(analysis.get_pc_ucm_angles, ucm_vec)
angle_df['user'] = pca_results['user']

# %%
angle_PC1_md = angle_df.loc[angle_df['PC']==1].groupby('task').agg('median')

# %%
fig_angles = px.histogram(angle_df.loc[angle_df['PC']==1, ['task', 'parallel']], barmode='overlay', nbins=20,
                          histnorm='percent', facet_row='task', opacity=0.7, height=600, labels={'task': "Task"})
fig_angles.update_yaxes(hoverformat='.2f', title="")
fig_angles.update_layout(showlegend=False, margin=plot.theme['graph_margins'],
                         xaxis_title="Interior Angle between PC1 and UCM (degrees)",
                         # keep the original annotations and add a list of new annotations:
                         annotations = list(fig_angles.layout.annotations) + 
                         [go.layout.Annotation(x=-0.07,
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
# ToDo: In plotly 4.12 we could add medians as vertical lines.

# %% [markdown]
# ## Save Reports

# %%
reports_path = Path.cwd() / 'reports'
# Save tables.
out_file = reports_path / 'pca-summary.csv'
var_expl_summary.to_csv(out_file, index=False)
logging.info(f"Written report to {out_file.resolve()}")

out_file = reports_path / 'pca-summary.tex'
var_expl_summary.to_latex(out_file, caption="Mean Explained Variance (\%) by Principal Components",
                          label="tab:PCA", float_format="%.2f")

out_file = reports_path / 'pca-ucm-angles.csv'
# Flatten multiindex for saving.
angle_PC1_md.to_csv(out_file, index=False)
logging.info(f"Written report to {out_file.resolve()}")

out_file = reports_path / 'pca-ucm-angles.tex'
angle_PC1_md.to_latex(out_file, caption="Median Interior Angle (deg) between UCM and PC1", label="tab:PCAAngle",
                      float_format="%.2f")

# Save figures.
figures_path = Path.cwd() / 'reports/figures'

fig_filepath = figures_path / 'barplot-pca.pdf'
pca_barplot.write_image(str(fig_filepath))
logging.info(f"Written figure to {fig_filepath.resolve()}")

fig_filepath = figures_path / 'histogram-PC1_UCM_angles.pdf'
fig_angles.write_image(str(fig_filepath))
logging.info(f"Written figure to {fig_filepath.resolve()}")
