#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
from pathlib import Path
from typing import Union
import warnings

import numpy as np
import pandas as pd
import patsy
import pymc3 as pm

from . import bridgesampler


class ModelComparison:
    """ Compare probabilities of models given data of projections onto vectors parallel and orthogonal to the UCM.
    
    * Model 0: All variances are equal in all blocks, but unknown. The null-model, no effect.
    * Model 1: Main effect of projection (parallel > orthogonal).
    * Model 2: Main effect of treatment (block 2 > block 1&3).
    * Model 3: Both main effects.
    * Model 4: Extraneous cognitive load / split-attention-effect leads to bad performance in constrained task.
    * Model 5: Synergy affects task performance by increasing precision.
    * Model 6: Synergy has no effect on task performance.
    * Model 7: Priming/Learning effect after constrained task.
    """

    def __init__(self, df_projections, min_samples=20):
        """
        :param df_projections: DataFrame with columns 'user', 'block', 'parallel', 'orthogonal'.
        :type df_projections: pandas.DataFrame
        :param min_samples: Minimum number of samples per block for each user.
                            Users will be excluded if they don't meet requirement.
                            As we don't partially pool across users, this threshold should prevent absurd estimations.
        :type min_samples: int
        """
        # Preprocess data.
        self.df = self.preprocess_data(df_projections, min_samples=min_samples)
        # Parameters for PyMC3 sampling method.
        # TODO: find good parameter values for sampling. Acceptance rate with NUTS should be around 0.8
        # Higher acceptance rate than 0.8 for proposed parameters is not better!
        # It just means we're not getting closer to a local maximum and are still climbing up the hill.
        self.sample_params = dict()
        self.model_funcs = [self.get_model_0,
                            self.get_model_1,
                            self.get_model_2,
                            self.get_model_3,
                            self.get_model_4,
                            self.get_model_5,
                            self.get_model_6,
                            self.get_model_7,
                            ]
        self.traces = {user: dict() for user in self.df['user'].unique()}
        self.models = {user: list() for user in self.df['user'].unique()}
        # ToDo: The priors are the shape and scale parameters, or mu and sigma of the Gamma distributions.
        # We test the hypotheses on each user and get a distribution of hypotheses distributions.
        self.posteriors = pd.DataFrame(columns=[f"M{i}" for i in range(len(self.model_funcs))],
                                       index=self.df['user'].unique())
        self.posteriors.rename_axis('user', inplace=True)

    def preprocess_data(self, dataframe, min_samples=20):  # Adjust min_samples per block
        """ Prepare incoming data for analysis.
        
        :param dataframe: Incoming data of projections.
        :type dataframe: pandas.DataFrame
        :param min_samples: Minimum number of samples per block for each user.
                            Users will be excluded if they don't meet requirement.
        :type min_samples: int
        :rtype: pandas.DataFrame
        """
        print("Preprocessing data...", end="")
        df = dataframe[['user', 'block', 'orthogonal', 'parallel']].copy()  # Keep only these columns.
        try:
            df[['user', 'block']] = df[['user', 'block']].astype('category')
            df.sort_values(by=['user', 'block'], inplace=True)
            # We need log-transformed squared deviations.
            # Alternatively use Box Cox transformation.
            df[['orthogonal', 'parallel']] = df[['orthogonal', 'parallel']].transform('square').transform('log')
        except KeyError:
            raise KeyError("Missing columns in dataframe. Make sure it has columns: "
                           "'user', 'block', 'parallel' and 'orthogonal'.")
        
        # If a user does not have enough samples in each block, exclude entirely.
        counts = df.groupby(['user', 'block']).agg(count=('block', 'count')).unstack(level='block')
        counts.columns = counts.columns.droplevel(0)
        mask = (counts >= min_samples).all(axis='columns')
        excluded = ', '.join([str(u) for u in mask[~mask].index.values])
        if excluded:
            warnings.warn(f"\Removing users {excluded} from analysis because they don't meet the requirement of "
                          f"having at least {min_samples} valid samples in all blocks.")
            df = df[df['user'].map(mask)]
        print("Done.")
        return df
    
    def get_coordinates(self, dataframe):
        """[summary]

        :param dataframe: [description]
        :type dataframe: [type]
        :return: [description]
        :rtype: [type]
        """
        # We need to be able to index by user. But the user IDs of the sample don't start at 0. We create an index and a mapping between them.
        users = dataframe['user'].cat.categories.tolist()
        # Block indices for 1,2,3.
        blocks = dataframe['block'].cat.categories.tolist()
        # Direction of projection.
        directions = ['orthogonal', 'parallel']
        coords = {'Direction': directions,
                  'User': users,
                  'Block': blocks,
                  'obs_id': np.arange(len(dataframe))}
        return coords
    
    def get_indices(self, dataframe):
        """[summary]

        :param dataframe: [description]
        :type dataframe: [type]
        """
        # Users
        user_indices = dataframe['user'].cat.codes.values  # User index for every single value.
        user_lookup = pd.Series(*pd.factorize(dataframe['user'].cat.categories), name='user_idx_lookup')
        # Blocks
        block_indices = dataframe['block'].cat.codes.values # Block index for every single value.
        block_lookup = pd.Series(*pd.factorize(dataframe['block'].cat.categories), name='block_idx_lookup')
        return user_indices, block_indices

    def get_block_dmatrix(self, dataframe):
        """[summary]

        :param dataframe: [description]
        :type dataframe: [type]
        :raises KeyError: [description]
        :return: [description]
        :rtype: [type]
        """
        block_mx = patsy.dmatrix('0 + block', dataframe, return_type='dataframe').astype(int)
        block_mx.columns = dataframe['block'].cat.categories.tolist()
        return block_mx

    def update_sample_params(self, model_name, draws=2000, tune=2000, target_accept=0.8):
        """[summary]

        :param model_name: [description]
        :type model_name: [type]
        :param draws: [description], defaults to 1000
        :type draws: int, optional
        :param tune: [description], defaults to 800
        :type tune: int, optional
        :param target_accept: [description], defaults to 0.8
        :type target_accept: float, optional
        """
        self.sample_params[model_name] = dict(draws=draws, tune=tune, target_accept=target_accept)

    #####################
    # Model Definitions #
    #####################
    def get_model_0(self, dataframe, prior_mu=(0.80, 2.74), prior_sigma=1.0):
        """ Compute marginal-log-likelihood of M0: all variances are equal in all blocks, but unknown. 
        For each user all data is generated by the same distribution.
        
        No synergy effects can be observed for this particular bi-manual task.
        
        Prior believe: Unlikely. Synergy effects have been observed time and time again.
        But thumbs could receive independent control signals as they're quite independent in the musculoskeletal system.
        
        :param prior_mu: Prior parameters for the total mean and its spread (standard error).
        :type prior_mu: tuple[float]
        :param prior_sigma: Prior parameters for the mean standard deviation. Lambda of exponential function.
        :type prior_sigma: float
        :type dataframe: pandas.DataFrame
        :rtype: float
        """
        coords = self.get_coordinates(dataframe)

        with pm.Model(coords=coords) as model:
            model.name = "M0"
            # Prior. non-centered.
            z = pm.Normal("z", mu=0, sigma=1)
            mu = pm.Deterministic("mu", prior_mu[0] + z * prior_mu[1])
            # Our coordinates aren't in long format, so we need to have the same prior 2 times to cover both directions.
            theta = pm.math.stack((mu, mu)).T
            # Model error.
            sigma = pm.Exponential("sigma", lam=prior_sigma)

            # Observed variable.
            projection_obs = pm.Data("projection_obs", dataframe[coords['Direction']], dims=('obs_id', 'Direction'))
            # Using user_idx to index theta somehow tells which prior belongs to which user.
            projection = pm.Normal("projection", mu=theta, sigma=sigma, observed=projection_obs,
                                   dims=('obs_id', 'Direction'))
        
        self.update_sample_params(model.name)  # defaults
        return model

    def get_model_1(self, dataframe, prior_mu_orthogonal=(0.09, 2.5), prior_diff_scale=2.5, prior_sigma=1.0):
        """ Compute marginal-log-likelihood of M5: Synergy is needed to perform a task.
        
        Main effect of projection (parallel > orthogonal).
        Strong synergies in all the tasks. This model implies a bad performance for the additional goal in the
        constrained task.
        
        Prior believe: Unlikely.
        
        :type dataframe: pandas.DataFrame
        :rtype: float
        """
        coords = self.get_coordinates(dataframe)

        with pm.Model(coords=coords) as model:
            model.name = "M1"
            # non-centered priors.
            z_ortho = pm.Normal('z_ortho', mu=0, sigma=1)
            mu_ortho = pm.Deterministic("mu_ortho", prior_mu_orthogonal[0] + z_ortho * prior_mu_orthogonal[1])
            # Assume positive difference.
            mu_diff = pm.HalfNormal('mu_diff', sigma=prior_diff_scale)
            mu_parallel = pm.Deterministic('mu_parallel', mu_ortho + mu_diff)

            # Stack priors.
            theta = pm.math.stack((mu_ortho, mu_parallel)).T
            # Model error:
            sigma = pm.Exponential("sigma", lam=prior_sigma, dims='Direction')
            # Observed variable.
            projection_obs = pm.Data("projection_obs", dataframe[coords['Direction']], dims=('obs_id', 'Direction'))
            # Using user_idx to index theta somehow tells which prior belongs to which user.
            projection = pm.Normal("projection", mu=theta, sigma=sigma, observed=projection_obs,
                                   dims=('obs_id', 'Direction'))

        self.update_sample_params(model.name)  # defaults
        return model

    def get_model_2(self, dataframe, prior_mu=(0.09, 2.5), prior_diff_scale=2.5, prior_sigma=1.0):
        """ Compute marginal-log-likelihood of M6:
        
        Main effect of treatment (block 2 > block 1&3).
        No synergy effects can be observed for this particular bi-manual task. The constrained task (block 2) appears
        too difficult to coordinate and results in heightened deviations in all directions compared to the
        unconstrained task (blocks 1&3).
        
        Prior believe: Unlikely
        
        :type dataframe: pandas.DataFrame
        :rtype: float
        """
        coords = self.get_coordinates(dataframe)
        block_mx = self.get_block_dmatrix(dataframe)

        with pm.Model(coords=coords) as model:
            model.name = "M2"
            block2_idx = pm.Data('block2_idx', block_mx[2].values, dims='obs_id')
            # Priors for blocks 1 an 3, non-centered.
            z_blocks13 = pm.Normal("z_blocks_1_3", mu=0, sigma=1)
            mu_blocks13 = pm.Deterministic('mu_blocks_1_3', prior_mu[0] + z_blocks13 * prior_mu[1])
            # Poisitive difference to Block 2.
            diff = pm.HalfNormal('mu_diff', sigma=prior_diff_scale)
            mu_block2 = pm.Deterministic('mu_block_2', mu_blocks13 + diff)

            # Expected deviation. block2_idx is either 0 or 1.
            theta_ = (1 - block2_idx) * mu_blocks13 + block2_idx * mu_block2
            theta = pm.math.stack((theta_, theta_)).T
            # Model error:
            sigma = pm.Exponential("sigma", lam=prior_sigma)
            # Observed variable.
            projection_obs = pm.Data("projection_obs", dataframe[coords['Direction']], dims=('obs_id', 'Direction'))
            # Using user_idx to index theta somehow tells which prior belongs to which user.
            projection = pm.Normal("projection", mu=theta, sigma=sigma, observed=projection_obs,
                                   dims=('obs_id', 'Direction'))
        
        self.update_sample_params(model.name, target_accept=0.9)
        return model

    def get_model_3(self, dataframe, 
                    prior_mu_ortho=(0.09, 2.5),
                    prior_diff_dir=2.5,
                    prior_diff_block=3.5,
                    prior_sigma=1.0):
        """ Compute marginal-log-likelihood of M4: Both main effects, for direction and block.
                                                   (Extraneous cognitive load I).
        
        Constrained task is perceived more difficult by visual instructions. Strong synergy in simpler task.
        
        Prior believe: Likely for users with no video-game experience. It's also likely that participants don't figure
        out the control scheme of df1=df2=target/2, since only one df seems constrained which might get more attention.
        
        :type dataframe: pandas.DataFrame
        :rtype: float
        """
        coords = self.get_coordinates(dataframe)
        block_mx = self.get_block_dmatrix(dataframe)

        with pm.Model(coords=coords) as model:
            model.name = "M3"
            block2_idx = pm.Data('block2_idx', block_mx[2].values, dims='obs_id')
            # Prior blocks 1 an 3 orthogonal, non-centered.
            z_blocks13_ortho = pm.Normal("z_blocks_1_3", mu=0, sigma=1)
            mu_blocks13_ortho = pm.Deterministic('mu_blocks_1_3_orthogonal',
                                                 prior_mu_ortho[0] + z_blocks13_ortho * prior_mu_ortho[1])

            # Poisitive differences. First for direction, second for block 2.
            diff = pm.HalfNormal('mu_diff',
                                 sigma=np.array([prior_diff_dir, prior_diff_block]),
                                 shape=2)
            
            mu_block2_ortho = pm.Deterministic('mu_block_2_orthogonal', mu_blocks13_ortho + diff[1])
            mu_blocks13_para = pm.Deterministic('mu_blocks_1_3_parallel', mu_blocks13_ortho + diff[0])
            mu_block2_para = pm.Deterministic('mu_block_2_parallel', mu_blocks13_ortho + diff[0] + diff[1])
            
            mu_ortho = (1 - block2_idx) * mu_blocks13_ortho + block2_idx * mu_block2_ortho
            mu_para = (1 - block2_idx) * mu_blocks13_para + block2_idx * mu_block2_para
            
            # Model error:
            sigma = pm.Exponential("sigma", lam=prior_sigma, dims='Direction')
            # Expected deviation per direction:
            theta = pm.math.stack((mu_ortho, mu_para)).T
            # Observed variable.
            projection_obs = pm.Data("projection_obs", dataframe[coords['Direction']], dims=('obs_id', 'Direction'))
            # Using user_idx to index theta somehow tells which prior belongs to which user.
            projection = pm.Normal("projection", mu=theta, sigma=sigma, observed=projection_obs,
                                   dims=('obs_id', 'Direction'))
        
        self.update_sample_params(model.name, target_accept=0.9)
        return model

    def get_model_4(self, dataframe, prior_mu=(0.09, 2.5), prior_diff_scale=2.5, prior_sigma=1.0):
        """ Compute marginal-log-likelihood of M4: Extraneous cognitive load / split-attention-effect.
        
        Constrained task is perceived more difficult by visual instructions. Strong synergy in simpler task.
        Parallel projection variability stays the same through all blocks. Orthogonal projection variability increases 
        in block 2.
        
        Prior believe: Likely for users with no video-game experience. It's also likely that participants don't figure
        out the control scheme of df1=df2=target/2, since only one df seems constrained which might get more attention.
        
        :type dataframe: pandas.DataFrame
        :rtype: float
        """
        coords = self.get_coordinates(dataframe)
        block_mx = self.get_block_dmatrix(dataframe)

        with pm.Model(coords=coords) as model:
            model.name = "M4"
            block2_idx = pm.Data('block2_idx', block_mx[2].values, dims='obs_id')
            # Blocks 1 an 3 orthogonal. Centered parameterization seems to cause less divergences?!
            mu_ortho = pm.Normal('mu_blocks_1_3_orthogonal', mu=prior_mu[0], sigma=prior_mu[1])
            # Poisitive difference to Block 2 AND parallel.
            diff = pm.HalfNormal('mu_diff', sigma=prior_diff_scale)
            mu_parallel = pm.Deterministic('mu_parallel', mu_ortho + diff)

            # Expected deviation per direction and block:
            theta_parallel = mu_parallel * np.ones(len(dataframe))  # Must match shape of orthogonal projections.
            theta_ortho = (1 - block2_idx) * mu_ortho + block2_idx * mu_parallel  # Raise block 2 orthogonal to parallel
            theta = pm.math.stack((theta_ortho, theta_parallel)).T
            
            # Model error:
            sigma = pm.Exponential("sigma", lam=prior_sigma, dims='Direction')

            # Observed variable.
            projection_obs = pm.Data("projection_obs", dataframe[coords['Direction']], dims=('obs_id', 'Direction'))
            # Using user_idx to index theta somehow tells which prior belongs to which user.
            projection = pm.Normal("projection", mu=theta, sigma=sigma, observed=projection_obs,
                                   dims=('obs_id', 'Direction'))

        self.update_sample_params(model.name, draws=2000, tune=3000, target_accept=0.8)
        return model

    def get_model_5(self, dataframe, 
                    prior_mu=(0.09, 2.5),
                    prior_mu_diff1=2.5,  # Difference between orthogonal projection block 1 and 3 to block 2.
                    prior_mu_diff2=2.5,  # Difference between orthogonal projection block 2 to parallel 1 and 3.
                    prior_sigma=1.0):
        """ Compute marginal-log-likelihood of M1: Synergy affects task performance by increasing precision.
        
        Following predictions made by Todorov (2004), in the constrained task (block 2) the variance in the direction
        parallel to the UCM can be reduced "at the expense of increased variance in the task-relevant direction"
        (orthogonal to UCM) in comparison to the unconstrained task (blocks 1&3).
        Parallel and orthogonal projection variances are expected to be roughly the same in the constrained task,
        since the control signals are expected to be the same.
        
        Prior believe: Likely. Synergy effect might stem from upper level in hierarchical control.
        Increased control signals in constrained task could lead to more muscle noise.
        
        :type dataframe: pandas.DataFrame
        :rtype: float
        """
        coords = self.get_coordinates(dataframe)
        block_mx = self.get_block_dmatrix(dataframe)

        with pm.Model(coords=coords) as model:
            model.name = "M5"
            block2_idx = pm.Data('block2_idx', block_mx[2].values, dims='obs_id')
            # Prior blocks 1 an 3 orthogonal, non-centered.
            z_blocks13_ortho = pm.Normal("z_blocks_1_3", mu=0, sigma=1)
            mu_blocks13_ortho = pm.Deterministic('mu_blocks_1_3_orthogonal',
                                                 prior_mu[0] + z_blocks13_ortho * prior_mu[1])
            # Poisitive differences. First for direction, second for block 2.
            diff = pm.HalfNormal('mu_diff',
                                 sigma=np.array([prior_mu_diff1, prior_mu_diff2]),
                                 shape=2)
            
            mu_block2 = pm.Deterministic('mu_block_2', mu_blocks13_ortho + diff[0])
            mu_blocks13_para = pm.Deterministic('mu_blocks_1_3_parallel', mu_blocks13_ortho + diff[0] + diff[1])
            # Variability in block 2 shouldbe the same for both directions.
            theta_ortho = (1 - block2_idx) * mu_blocks13_ortho + block2_idx * mu_block2
            theta_para = (1 - block2_idx) * mu_blocks13_para + block2_idx * mu_block2
            
            # Model error:
            sigma = pm.Exponential("sigma", lam=prior_sigma, dims='Direction')
            # Expected deviation per direction:
            theta = pm.math.stack((theta_ortho, theta_para)).T
            # Observed variable.
            projection_obs = pm.Data("projection_obs", dataframe[coords['Direction']], dims=('obs_id', 'Direction'))
            # Using user_idx to index theta somehow tells which prior belongs to which user.
            projection = pm.Normal("projection", mu=theta, sigma=sigma, observed=projection_obs,
                                   dims=('obs_id', 'Direction'))
        
        self.update_sample_params(model.name, tune=3000, target_accept=0.9)
        return model
    
    def get_model_6(self, dataframe, prior_mu=(0.09, 2.5), prior_diff_scale=2.5, prior_sigma=1.0):
        """ Compute marginal-log-likelihood of M2: Synergy has no effect on task performance.
        
        Orthogonal variances are small in all blocks.
        Danion, F., & Latash, M. L. (2011): Same performance variance without synergy in constrained task (block 2).
        Parallel deviations in block 2 are on average as large as orthogonal deviations in all blocks,
        and larger in blocks 1&3 (because they are uncontrolled).
        
        Prior believe: Likely. A thumbs' movement does not directly influence the other through the
        musculoskeletal system (but maybe through the device connecting the two). We just haven't exhausted our control
        over all degrees of freedom yet in the unconstrained task, but when necessary in the constrained task, we can
        do it.
        Even increased control signals could be so low that we fail to detect the effect of increased muscle noise.
        
        :type dataframe: pandas.DataFrame
        :rtype: float
        """
        coords = self.get_coordinates(dataframe)
        block_mx = self.get_block_dmatrix(dataframe)

        with pm.Model(coords=coords) as model:
            model.name = "M6"
            block2_idx = pm.Data('block2_idx', block_mx[2].values, dims='obs_id')
            # Prior orthogonal projections, non-centered.
            z_ortho = pm.Normal("z_ortho", mu=0, sigma=1)
            mu_ortho = pm.Normal('mu_orthogonal', prior_mu[0] + z_ortho * prior_mu[1])
            # Poisitive difference to parallel projections in blocks 1 and 3.
            diff = pm.HalfNormal('mu_diff', sigma=prior_diff_scale)
            mu_parallel_blocks13 = pm.Deterministic('mu_parallel_blocks_1_3', mu_ortho + diff)

            # Expected deviation per direction and block:
            theta_ortho = mu_ortho * np.ones(len(dataframe))  # Must be same length as theta_parallel for stacking.
            theta_parallel = block2_idx * mu_ortho + (1 - block2_idx) * mu_parallel_blocks13
            theta = pm.math.stack((theta_ortho, theta_parallel)).T
            
            # Model error:
            sigma = pm.Exponential("sigma", lam=prior_sigma, dims='Direction')

            # Observed variable.
            projection_obs = pm.Data("projection_obs", dataframe[coords['Direction']], dims=('obs_id', 'Direction'))
            # Using user_idx to index theta somehow tells which prior belongs to which user.
            projection = pm.Normal("projection", mu=theta, sigma=sigma, observed=projection_obs,
                                   dims=('obs_id', 'Direction'))

        self.update_sample_params(model.name, tune=3000, target_accept=0.9)
        return model

    def get_model_7(self, dataframe, 
                    prior_mu=(0.09, 2.5),
                    prior_mu_diff1=2.5,  # Difference between orthogonal projection block 1 and 3 to block 2.
                    prior_mu_diff2=2.5,  # Difference between orthogonal projection block 2 to parallel 1.
                    prior_sigma=1.0):
        """ Compute marginal-log-likelihood of M3: Priming/Learning effect after constrained task.
        
        Strong synergy in first unconstrained task, no synergy in constrained task, weak/no synergy in the
        unconstrained task. With optimal control we'd expect strong synergies again as soon as constrains are lifted.
        This model contradicts the prediction and postulates a priming effect of the constrained task onto the
        following unconstrained task.
        # ToDo: insert reference on reduced synergy with higher precision through training.
        
        Prior believe: Likely
        
        :type dataframe: pandas.DataFrame
        :rtype: float
        """
        coords = self.get_coordinates(dataframe)
        block_mx = self.get_block_dmatrix(dataframe)

        with pm.Model(coords=coords) as model:
            model.name = "M7"
            blocks_idx = pm.Data('blocks_idx', block_mx.values, dims=('obs_id', 'Block'))
            # Prior blocks 1 an 3 orthogonal, non-centered.
            z_blocks13_ortho = pm.Normal("z_blocks_1_3", mu=0, sigma=1)
            mu_blocks13_ortho = pm.Deterministic('mu_blocks_1_3_orthogonal',
                                                 prior_mu[0] + z_blocks13_ortho * prior_mu[1])
            # Poisitive differences. First for direction, second for block 2.
            diff = pm.HalfNormal('mu_diff',
                                 sigma=np.array([prior_mu_diff1, prior_mu_diff2]),
                                 shape=2)
            
            mu_block2 = pm.Deterministic('mu_block_2', mu_blocks13_ortho + diff[0])
            mu_block1_para = pm.Deterministic('mu_block_1_parallel', mu_blocks13_ortho + diff[0] + diff[1])
            # Variability in block 2 shouldbe the same for both directions.
            theta_ortho = (1 - blocks_idx[:, 1]) * mu_blocks13_ortho + blocks_idx[:, 1] * mu_block2
            theta_para = blocks_idx[:, 0] * mu_block1_para + (1 - blocks_idx[:, 0]) * mu_block2
            
            # Model error:
            sigma = pm.Exponential("sigma", lam=prior_sigma, dims='Direction')
            # Expected deviation per direction:
            theta = pm.math.stack((theta_ortho, theta_para)).T
            # Observed variable.
            projection_obs = pm.Data("projection_obs", dataframe[coords['Direction']], dims=('obs_id', 'Direction'))
            # Using user_idx to index theta somehow tells which prior belongs to which user.
            projection = pm.Normal("projection", mu=theta, sigma=sigma, observed=projection_obs,
                                   dims=('obs_id', 'Direction'))
        
        self.update_sample_params(model.name, tune=3000, target_accept=0.9)
        return model

    def get_models(self, data):
        """ Get a list with all models.
        
        :param data: Observed squared projections for each block. Needs columns 'block', 'parallel', 'orthogonal'.
        :type data: pandas.DataFrame
        :return: Models
        :rtype: list[pymc3.Model]
        """
        models = list()
        print(f"Building {len(self.model_funcs)} models...")
        for i, func in enumerate(self.model_funcs):
            model = func(data)
            models.append(model)
            print(f"Model {i}: '{model.name}' done.")
        return models
    
    #############
    # Inference #
    #############
    def sample(self, model):
        """ Inference by sampling posterior.

        :type model: pymc3.Model
        :rtype trace: Union[pymc3.backends.base.MultiTrace, arviz.InferenceData]
        """
        with model:
            # Sample.
            print(f"\nSampling for Model: {model.name}.")
            trace = pm.sample(self.sample_params[model.name]['draws'],
                              tune=self.sample_params[model.name]['tune'],
                              target_accept=self.sample_params[model.name]['target_accept'])
        return trace
        
    def get_marginal_likelihood(self, model, trace=None):
        """ Bridge sampling.
        
        :type model: pymc3.Model
        :type trace: Union[pymc3.backends.base.MultiTrace, arviz.InferenceData]
        :rtype: float
        """
        if trace is None:
            trace = self.sample(model)
        with model:
            # Estimate marginal log-likelihood with bridge sampling.
            print(f"Computing marginal log-likelihood of Model|Data for model: {model.name}")
            marginal_log_likelihood = bridgesampler.marginal_llk(trace, model=model, maxiter=10000)["logml"]
        return marginal_log_likelihood
        
    def _filter_by_user(self, user):
        """ Get subset of data for specific user.
        
        :param user: Identifier for user in data.
        :type user: int
        :rtype: pandas.DataFrame
        """
        if user not in self.df['user'].cat.categories:
            raise KeyError(f"User {user} not found in data set.")
        return self.df[self.df['user'] == user]
    
    def compare_models(self, user):
        """ Compute model posterior for data of a specific user.

        :param user: For which user model likelihoods should be computed.
        :type user: int
        :return: Dictionary holding for each model its posterior probability.
        :rtype: dict[int]
        """
        print(f"\nCommencing model comparison for user {user}...\n")
        # Get models for observed data of participant.
        df = self._filter_by_user(user)
        # TODO: Create models first, then just set new data on existing models?
        models = self.models[user] = self.get_models(df)
        # First, sample all models.
        for model in models:
            self.traces[user].update({model.name: self.sample(model)})
        # Secondly, compute all marginal log-likelihoods,
        model_posterior = np.array([self.get_marginal_likelihood(model, trace=self.traces[user][model.name]) 
                                    for model in models])
        # then, exponentiate and normalize to turn into a probability distribution.
        model_posterior = np.exp(model_posterior-np.logaddexp.reduce(model_posterior))
        # Save posterior for this participant.
        self.posteriors.loc[user] = model_posterior
        return {i: mp for i, mp in enumerate(model_posterior)}

    def write_traces(self, destination, user=None):
        """ Create directory for each user. In each user directory create folders for each model.
        Each chain goes inside a directory, and each directory contains a metadata json file,
        and a numpy compressed file.

        :param destination: Directory to save to.
        :type destination: Union[str, Path]
        :param user: Only write traces of this user.
        :type: int
        """
        if isinstance(destination, str):
            destination = Path(destination)

        if user:
            if user not in self.traces.keys():
                warnings.warn(f"User {user} not found!")
                return
            elif not self.traces[user]:
                warnings.warn(f"No traces for user {user}!")
                return

        # ToDo: parallelize.
        print("Writing traces to files... ")
        for u, models in self.traces.items():
            # If user given, only process this one.
            if user and u!=user:
                continue
            # User folder.
            user_folder = destination / f'user_{u}'
            for model_name, trace in models.items():
                # Create model folder.
                model_folder = user_folder / model_name
                model_folder.mkdir(parents=True, exist_ok=True)
                pm.save_trace(trace, directory=str(model_folder), overwrite=True)
        print("Done.\n")

    def write_posteriors(self, destination):
        """ Write posterior distribution to file.

        :param destination: CSV file to write to. Will overwrite.
        :type destination: pathlib.Path
        """
        if self.posteriors.isna().all().all():
            print(f"ERROR: Posteriors not yet computed. File {destination} not written.")
            return

        print("Writing posteriors to file: ", destination)
        self.posteriors.dropna(how='all').to_csv(destination)
