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
import plotly.io as pio

from neuropsymodelcomparison.dataprocessing import analysis
from neuropsymodelcomparison import plot

# Default file format for figures.
pio.kaleido.scope.default_format = "pdf"

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# %% [markdown]
# Get preprocessed trials data.

# %%
data_path = Path.cwd() / 'data/preprocessed/trials.csv'
df = pd.read_csv(data_path, index_col='id')
# Easier on memory and faster groupby.
df[['user', 'session', 'block', 'block_id', 'condition', 'task']] = df[['user', 'session', 'block', 'block_id',
                                                                        'condition', 'task']].astype('category')

# When we view statistics by task, we want them to to be displyed in a certain order.
task_display_order = ['pre', 'df1', 'df2', 'df1|df2', 'post']

# %% [markdown]
# ## PCA across participants

# %%
pca_results = df.groupby('task').apply(analysis.get_pca_data).loc[task_display_order]
pca_results.reset_index(inplace=True)
pca_results[['task', 'PC']] = pca_results[['task', 'PC']].astype('category')
pca_barplot = plot.generate_pca_figure(pca_results)

# %% [markdown]
# ## Visualize Principal Components

# %%
fig_scatter = plot.generate_trials_figure(df, marker_opacity=0.4, marker_size=6, width=500)
pc_arrows = plot.get_pca_annotations(pca_results)
fig_scatter.layout.update(annotations=pc_arrows)
plot.add_pca_ellipses(fig_scatter, pca_results)

# %% [markdown]
# Arrows display the direction and relative magnitude of the principal components. 
# The length of the arrows represent 3 times the square root of explained variance, 
# in other terms the explained standard deviation.
# The sizes of the semi-major and semi-minor axes of the ellipses are 2 times the 
# square root of explained variance by the respective principal components. 
# 
# %% [markdown]
# ## Differences between Principal Components and Uncontrolled Manifold
# We measure the differences between the directions of the principal components and the vectors parallel and orthogonal to the UCM in degrees.

# %%
ucm_vec = analysis.get_ucm_vec()
angle_df = pca_results.groupby('task', observed=True, sort=False).apply(analysis.get_pc_ucm_angles, ucm_vec)

# %% [markdown]
# ## Save Reports

# %%
reports_path = Path.cwd() / 'reports'
# Save tables.
out_file = reports_path / 'pca-results.csv'
pca_results.to_csv(out_file, index=False)
logging.info(f"Written report to {out_file.resolve()}")

out_file = reports_path / 'pca-ucm-angles.csv'
# Flatten multiindex for saving.
angle_df.to_csv(out_file, index=False)
logging.info(f"Written report to {out_file.resolve()}")

# Save figures.
figures_path = Path.cwd() / 'reports/figures'

barplot_filepath = figures_path / 'pca-barplot.pdf'
pca_barplot.write_image(str(barplot_filepath))
logging.info(f"Written figure to {barplot_filepath.resolve()}")

fig_scatter_filepath = figures_path / 'pca_scatter.pdf'
fig_scatter.write_image(str(fig_scatter_filepath))
logging.info(f"Written figure to {fig_scatter_filepath.resolve()}")
