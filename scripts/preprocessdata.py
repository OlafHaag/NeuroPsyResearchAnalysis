# notebook: 00-oh-preprocess_data.ipynb

# %% [markdown]
# # Data Cleanup and Pre-processing
# 
# Before we can analyze the data we need to clean the raw data and bring it to a format suited for the analyses.

# %%
# Basic imports and setup.

import sys
import logging
from pathlib import Path

import pandas as pd

from neuropsymodelcomparison.dataprocessing import analysis

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# %% [markdown]
# Read in raw data from files.

# %%
data_path = Path.cwd() / 'data'  # Run from project root.
raw_data_path = data_path / 'raw'

users = pd.read_csv(raw_data_path / 'users.csv', dtype={'gaming_exp': pd.Int8Dtype()})
blocks = pd.read_csv(raw_data_path / 'blocks.csv', index_col='id', parse_dates=['time_iso'])
trials = pd.read_csv(raw_data_path / 'trials.csv', index_col='id')

# %% [markdown]
# Data collection began on August 1st 2020 and ended on August 31st 2020. Data collected before that time period was 
# pilot data to test the funtionality of the app and tune the priors.  
# 

# %%
start_date = '2020-08-01'
end_date = '2020-09-01'  # At the beginning of the day at 00:00:00, so at the end of the previous day.

# How many testers were there?
n_testers = blocks.loc[blocks['time_iso'] < start_date, 'user_id'].nunique()
# We're only interested in first time participations.
blocks = blocks.loc[blocks['nth_session'] == 1, :]
# Keep only blocks within data collection time period.
blocks = blocks.loc[(blocks['time_iso'] > start_date) & (blocks['time_iso'] < end_date), :]
# How many people took part during that period?
n_users = blocks['user_id'].nunique()

# Keep only those users that participated in the given time period.
users = users.loc[users['id'].isin(blocks['user_id'].unique()), :]

# %% [markdown]
# Filter out invalid data. Keep a record of what was excluded and for which reasons.

# %%
# If a subsequent block is completed within 2 seconds after the previous one, there was a malfunction in the app. 
# A session must consist of 3 blocks.
blocks, n_errors, invalid_sessions = analysis.remove_erroneous_blocks(blocks, delta_time=2.0, n_blocks=3)
n_users_malfunction = n_users - blocks['user_id'].nunique()
# Merge data to 1 table.
df = analysis.join_data(users, blocks, trials)
# Remove trials where sliders where not grabbed concurrently or grabbed at all.
df, n_trials_removed = analysis.get_valid_trials(df)
# Further remove trials for which sliders where grabbed with too much time apart.
# The arbitrary choice for a threshold is set to a third of the available time.
n_trials = len(df)
df = df.loc[df['grab_diff'] < (blocks['trial_duration'].median()/3), :]
n_trials_removed += n_trials - len(df)

# %% [markdown]
# Save intermediate data to file.

# %%
interim_data_path = data_path / 'interim'
# Save trial data.
file_path = interim_data_path / 'trials.csv'
df.to_csv(file_path)
logging.info(f"Written interim trials data to {file_path.resolve()}")

# Save data about exclusions.
sampling_path = interim_data_path / 'sampling.txt'
with sampling_path.open(mode='w') as f:
    f.write(f"start_date={start_date}\n")
    f.write(f"end_date={str((pd.to_datetime(end_date) - pd.to_timedelta(1, unit='d')).date())}\n")
    f.write(f"n_testers={n_testers}\n")
    f.write(f"n_users={n_users}\n")
    f.writelines([f"total_gender_{k}={v}\n" for k,v in users['gender'].value_counts(dropna=False).iteritems()])
    f.writelines([f"total_age_{k}={v}\n" for k,v in users['age_group'].value_counts(dropna=False).iteritems()])
    f.write(f"n_excluded_malfunction={n_users_malfunction}\n")
    f.write(f"n_invalid_sessions={len(invalid_sessions)}\n")
    f.write(f"n_invalid_trials={n_trials_removed}\n")
logging.info(f"Written sampling data to {sampling_path.resolve()}")
