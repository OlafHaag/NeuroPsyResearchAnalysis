from pathlib import Path
import pandas as pd

import neuropsymodelcomparison as npmc


if __name__ == "__main__":
    import logging
    import sys

    # Retrieve data from database.
    devices = npmc.dataio.get_db_table('devices')
    users = npmc.dataio.get_db_table('users')
    blocks = npmc.dataio.get_db_table('circletask_blocks')
    trials = npmc.dataio.get_db_table('circletask_trials')

    # When we didn't collect all data, stop here.
    if True in [df.empty for df in [devices, users, blocks, trials]]:
        logging.error("Could not retrieve all raw data from database.")
        sys.exit(1)

    # Save to files.
    raw_data_folder = Path.cwd() / "data/raw"

    devices.to_csv(raw_data_folder / 'devices.csv')
    users.to_csv(raw_data_folder / 'users.csv')
    blocks.to_csv(raw_data_folder / 'blocks.csv')
    trials.to_csv(raw_data_folder / 'trials.csv')
