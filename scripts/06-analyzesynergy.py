# notebook source: 0-oh-analyze_synergy.ipynb

# %% [markdown]
# # Analysis of Synergy Indices
# ## A frequentist approach.
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
import plotly.io as pio
import seaborn as sns

from neuropsymodelcomparison.dataprocessing import analysis
from neuropsymodelcomparison import plot

sns.set(style="darkgrid")

# Default file format for plotly figures.
pio.kaleido.scope.default_format = "pdf"

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# %% [markdown]
# Get preprocessed trials data.

# %%
data_path = Path.cwd() /'reports/block_stats.csv'
figures_path = Path.cwd() /'reports/figures'

df = pd.read_csv(data_path, index_col='block_id', dtype={'exclude': bool})
df = df.loc[~df['exclude']].drop('exclude', axis='columns')

# Easier on memory and faster groupby.
df[['user', 'session', 'block', 'condition', 'task']] = df[['user', 'session', 'block', 
                                                            'condition', 'task']].astype('category')

# When we view statistics by task, we want them to to be displyed in a certain order.
task_display_order = ['pre', 'df1', 'df2', 'df1|df2', 'post']
df.task.cat.reorder_categories(task_display_order, inplace=True)
condition_display_order = ['df1', 'df2', 'df1|df2']
df.condition.cat.reorder_categories(condition_display_order, inplace=True)

# %% [markdown]
# ## Projection variance per direction to UCM
# 
# ###  For each participant

# %%
df_long = analysis.wide_to_long(df, ['parallel', 'orthogonal'], suffixes='variance', j='projection')
fig_proj_line = plot.generate_lines_plot(df_long, "variance", by='user', color_col='projection',
                                         category_orders={'condition': condition_display_order}, width=1000)

# %% [markdown]
# ### Across participants

# %%
fig_proj_line2 = sns.catplot(x="Block", y="Variance", hue="Projection", col="condition", 
                             col_order=condition_display_order,
                             data=df_long.rename(columns={'variance': 'Variance', 
                                                          'block': 'Block',
                                                          'projection': 'Projection'}),
                             kind="point", dodge=True, width=10, aspect=.7,
                             palette={'parallel': 'lightgreen', 'orthogonal': 'salmon'})

fig_filepath = figures_path / 'line-plot-projections.pdf'
plt.savefig(str(fig_filepath))
logging.info(f"Written figure to {fig_filepath.resolve()}")


# %%
fig_proj_violin = plot.generate_violin_figure(
                        df.rename(columns={'parallel variance': 'parallel', 'orthogonal variance': 'orthogonal'}),
                        columns=['parallel', 'orthogonal'], ytitle='Variance', legend_title="PROJECTION", width=1000
                                             )

# %% [markdown]
# ## Inferential Statistics
# ### Wilcoxon Rank Test

# %%
# Overall comparison.
decision, w, p = analysis.wilcoxon_rank_test(
                            df.rename(columns={'parallel variance': 'parallel', 'orthogonal variance': 'orthogonal'})
                                            )

wilcoxon_results = pd.DataFrame({'decision': decision, 'test statistic': w, 'p': p},
                                index=pd.MultiIndex.from_tuples([('overall', '')], names=('condition', 'task')))
# Per condition and task.
# ToDo: Correct p-values for multiple testing?
wilcoxon_by_condition = df.groupby(['condition', 'task'])\
                          .apply(lambda x: analysis.wilcoxon_rank_test(x.rename(columns={
                                                                                    'parallel variance': 'parallel',
                                                                                    'orthogonal variance': 'orthogonal'
                                                                                        }
                                                                               )
                                                                      )
                                ).apply(pd.Series).rename(columns={0: 'decision', 1: 'test statistic', 2: 'p'})
wilcoxon_results = wilcoxon_results.append(wilcoxon_by_condition)

# %% [markdown]
# ## Visual inspection of Synergy Index

# %%
fig_dVz = sns.catplot(x="block", y="$\\Delta V_z$",
                      hue="user", col="condition",
                      data=df.rename(columns={'dVz': '$\\Delta V_z$'}), kind="point",
                      dodge=True,
                      height=4, aspect=.7)

fig_filepath = figures_path / 'line-plot-dVz_by_user.pdf'
plt.savefig(str(fig_filepath))
logging.info(f"Written figure to {fig_filepath.resolve()}")


# %%
fig_dVz = sns.catplot(x="block", y="$\\Delta V_z$",
                      hue="condition",
                      data=df.rename(columns={"dVz": "$\\Delta V_z$"}), kind="point",
                      dodge=True,
                      height=4, aspect=1.333)

fig_filepath = figures_path / 'line-plot-dVz.pdf'
plt.savefig(str(fig_filepath))
logging.info(f"Written figure to {fig_filepath.resolve()}")

# %% [markdown]
# ### Normality of Fisher-z-transformed Synergy Index
# Make sure the tranformation worked.

# %%
norm_dVz = df.groupby('task')['dVz'].apply(lambda x: pg.normality(x).iloc[0]).unstack(level=1)

# %% [markdown]
# ### Mixed ANOVA
# 

# %%
anova_dVz = analysis.mixed_anova_synergy_index_z(df)

# %% [markdown]
# ## Posthoc Testing

# %%
posthoc_comparisons = analysis.posthoc_ttests(df)

# %%
dVz_export = df[['user', 'condition', 'block', 'dVz']]\
             .set_index(['user', 'condition', 'block'])\
             .unstack(level='block')\
             .reset_index(level='condition')
dVz_export.columns = [f"{dVz_export.columns.names[1]}{col}" if col else "condition" 
                      for col in dVz_export.columns.get_level_values('block')]

# %% [markdown]
# ### Save Reports

# %%
reports_path = Path.cwd() /'reports'
# Save tables.
out_file = reports_path / 'wilcoxon-projection-results.csv'
wilcoxon_results.reset_index().to_csv(out_file, index=False)
logging.info(f"Written report to {out_file.resolve()}")

out_file = reports_path / 'normality-test-dVz.csv'
norm_dVz.to_csv(out_file)
logging.info(f"Written report to {out_file.resolve()}")

out_file = reports_path / 'anova-dVz.csv'
anova_dVz.to_csv(out_file, index=False)
logging.info(f"Written report to {out_file.resolve()}")

out_file = reports_path / 'posthoc-dVz.csv'
posthoc_comparisons.to_csv(out_file, index=False)
logging.info(f"Written report to {out_file.resolve()}")

out_file = reports_path / 'dVz-export.csv'
dVz_export.to_csv(out_file)
logging.info(f"Written report to {out_file.resolve()}")

# Save figures.
fig_filepath = figures_path / 'line-plot-projections_by_user.pdf'
fig_proj_line.write_image(str(fig_filepath))
logging.info(f"Written figure to {fig_filepath.resolve()}")

fig_filepath = figures_path / 'violin-projections.pdf'
fig_proj_violin.write_image(str(fig_filepath))
logging.info(f"Written figure to {fig_filepath.resolve()}")


