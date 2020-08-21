from pathlib import Path
import pandas as pd

import neuropsymodelcomparison as npmc


if __name__ == "__main__":
    # Retrieve data from database.
    devices = npmc.dataio.get_db_table('devices')
    users = npmc.dataio.get_db_table('users')
    blocks = npmc.dataio.get_db_table('circletask_blocks')
    trials = npmc.dataio.get_db_table('circletask_trials')

    # Save to files.
    raw_data_folder = Path.cwd() / "data/raw"

    devices.to_csv(raw_data_folder / 'devices.csv')
    users.to_csv(raw_data_folder / 'users.csv')
    blocks.to_csv(raw_data_folder / 'blocks.csv')
    trials.to_csv(raw_data_folder / 'trials.csv')
