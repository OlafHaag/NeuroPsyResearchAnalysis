from pathlib import Path
import pytest
import pandas as pd
import numpy as np

import neuropsymodelcomparison as npmc


class TestModelComparator:

    def create_dummy_data(self):
        # Todo: break down into single model dummy data to test each separately.
        # ++++++++++++++++++++++++++++++++++++
        # ++ Model 0 conform data           ++
        # ++++++++++++++++++++++++++++++++++++
        # Create some data conforming to M0: all deviations on average equal across blocks.
        n_trials = 270 - np.random.randint(36)
        df0 = pd.DataFrame({'user': 0, 'block': 1 + np.random.randint(3, size=n_trials),
                            'parallel': np.random.normal(0.0, 4.0, size=n_trials),
                            'orthogonal': np.random.normal(0.0, 4.0, size=n_trials)})
        
        # ++++++++++++++++++++++++++++++++++++
        # ++ Model 1 conform data           ++
        # ++++++++++++++++++++++++++++++++++++
        # Create some data conforming to M1.
        # Variable length of trials per block due to exclusions.
        n_trials = 30 - np.random.randint(12, size=3)
        # blocks 1&3 have smallest orthogonal deviations, but largest parallel deviations. Strong synergy.
        block1 = pd.DataFrame({'user': 1, 'block': 1,
                            'parallel': np.random.normal(0.0, 3.0, size=n_trials[0]),
                            'orthogonal': np.random.normal(0.0, 1.0, size=n_trials[0])})
        block3 = pd.DataFrame({'user': 1, 'block': 3,
                            'parallel': np.random.normal(0.0, 3.0, size=n_trials[1]),
                            'orthogonal': np.random.normal(0.0, 1.0, size=n_trials[1])})
        # Block 2 has smaller parallel deviations than blocks 1&3, but higher orthogonal deviations than blocks 1&3.
        # No synergy.
        block2 = pd.DataFrame({'user': 1, 'block': 2,
                            'parallel': np.random.normal(0.0, 2.0, size=n_trials[2]),
                            'orthogonal': np.random.normal(0.0, 2.0, size=n_trials[2])})
        df1 = pd.concat((block1, block2, block3), axis='index')

        # ++++++++++++++++++++++++++++++++++++
        # ++ Model 2 conform data           ++
        # ++++++++++++++++++++++++++++++++++++
        # Create some data conforming to M1.
        # Variable length of trials per block due to exclusions.
        n_trials = 30 - np.random.randint(12, size=3)
        # blocks 1&3 have smallest orthogonal deviations, but largest parallel deviations. Strong synergy.
        block1 = pd.DataFrame({'user': 2, 'block': 1,
                            'parallel': np.random.normal(0.0, 3.0, size=n_trials[0]),
                            'orthogonal': np.random.normal(0.0, 1.0, size=n_trials[0])})
        block3 = pd.DataFrame({'user': 2, 'block': 3,
                            'parallel': np.random.normal(0.0, 3.0, size=n_trials[1]),
                            'orthogonal': np.random.normal(0.0, 1.0, size=n_trials[1])})
        # Block 2 has smaller parallel deviations than in blocks 1&3, equal to orthogonal deviations in blocks in average.
        # No synergy.
        block2 = pd.DataFrame({'user': 2, 'block': 2,
                            'parallel': np.random.normal(0.0, 1.0, size=n_trials[2]),
                            'orthogonal': np.random.normal(0.0, 1.0, size=n_trials[2])})
        df2 = pd.concat((block1, block2, block3), axis='index')
        # Have one dataset with each user representing a model.
        df = pd.concat((df0, df1, df2), axis='index')
        return df
    
    @pytest.fixture(scope='class')
    def comparison(self):
        df = self.create_dummy_data()
        # ++++++++++++++++++++++++++++++++++++
        # ++ Compute posteriors             ++
        # ++++++++++++++++++++++++++++++++++++
        model_comp = npmc.dataprocessing.ModelComparison(df, min_samples=10)
        # Compute model posterior for each user.
        for user in range(3):
            print(f"Validate Model {user} Calculations.")
            model_comp.compare_models(user=user)

        return model_comp

    def test_non_empty_posterior(self, comparison):
        assert not comparison.posteriors.isna().all().all()

    def test_posterior_write(self, comparison):
        # Write results to file.
        output_file = Path.cwd() / "reports/tests/testdata_posteriors.csv"
        comparison.write_posteriors(output_file)
        assert output_file.is_file()
            