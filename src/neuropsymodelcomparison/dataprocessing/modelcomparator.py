#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
from pathlib import Path
from typing import Union

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm

from . import bridgesampler


class ModelComparison:
    """ Compare probabilities of models given data of projections onto vectors parallel and orthogonal to the UCM.
    
    * Model 0: All variances are equal in all blocks, but unknown. The null-model, no effect.
    * Model 1: Synergy affects task performance by increasing precision.
    * Model 2: Synergy has no effect on task performance.
    * Model 3: Priming/Learning effect after constrained task.
    * Model 4: Extraneous cognitive load / split-attention-effect leads to bad performance in constrained task.
    * Model 5: Main effect of projection (parallel > orthogonal).
    * Model 6: Main effect of treatment (block 2 > block 1&3).
    """

    def __init__(self, df_projections, min_samples=20):
        """
        :param df_projections: DataFrame with columns 'user', 'block', 'parallel', 'orthogonal'.
        :type df_projections: pandas.DataFrame
        :param min_samples: Minimum number of samples per block for each user.
                            Users will be excluded if they don't meet requirement.
        :type min_samples: int
        """
        # Preprocess data.
        self.df = self.preprocess_data(df_projections, min_samples=min_samples)
        # Parameters for PyMC3 sampling method.
        # TODO: find good parameter values for sampling. Acceptance rate with NUTS should be around 0.8
        # Higher acceptance rate than 0.8 for proposed parameters is not better!
        # It just means we're not getting closer to a local maximum and are still climbing up the hill.
        self.draws = 2000
        self.tune = 2000
        self.model_funcs = [self.get_model_0,
                            self.get_model_1,
                            self.get_model_2,
                            self.get_model_3,
                            self.get_model_4,
                            self.get_model_5,
                            self.get_model_6,
                            ]
        self.traces = {user: dict() for user in self.df['user'].unique()}
        # ToDo: The priors are the shape and scale parameters, or mu and sigma of the Gamma distributions.
        # We test the hypotheses on each user and get a distribution of hypotheses distributions.
        self.posteriors = pd.DataFrame(columns=[f"M{i}" for i in range(len(self.model_funcs))],
                                       index=self.df['user'].unique())
        self.posteriors.rename_axis('user', inplace=True)
        # ToDo: Consider participant's experience and task difficulty ratings.

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
        df = dataframe.copy()
        try:
            df[['user', 'block']] = df[['user', 'block']].astype('category')
            df.sort_values(by=['user', 'block'], inplace=True)
            # We need squared deviations.
            df[['parallel', 'orthogonal']] = df[['parallel', 'orthogonal']].transform('square')
        except KeyError:
            raise KeyError("Missing columns in dataframe. Make sure it has columns: "
                           "'user', 'block', 'parallel' and 'orthogonal'.")
        
        # If a user does not have enough samples in each block, exclude entirely.
        counts = df.groupby(['user', 'block']).agg(count=('block', 'count')).unstack(level='block')
        counts.columns = counts.columns.droplevel(0)
        mask = (counts >= min_samples).all(axis='columns')
        excluded = ', '.join([str(u) for u in mask[~mask].index.values])
        if excluded:
            print(f"\nWARNING: Removing users {excluded} from analysis because they don't meet the requirement of "
                  f"having at least {min_samples} valid samples in all blocks.")
            df = df[df['user'].map(mask)]
        print("Done.")
        return df
    
    #####################
    # Model Definitions #
    #####################
    def get_model_0(self, dataframe):
        """ Compute marginal-log-likelihood of M0: all variances are equal in all blocks, but unknown.
        
        No synergy effects can be observed for this particular bi-manual task.
        
        Prior believe: Unlikely. Synergy effects have been observed time and time again.
        But thumbs could receive independent control signals as they're quite independent in the musculoskeletal system.
        
        :type dataframe: pandas.DataFrame
        :rtype: float
        """
        with pm.Model() as model:
            model.name = "Null Model"
            # Assume that the squared deviations are drawn from a Gamma distribution with unknown parameters.
            # Both parameters have to be positive, so draw them from a Gamma prior, too.
            # TODO: Perhaps use the alternative parametrization with mu, sigma instead, once you have the pilot data.
            mu = pm.Gamma("mu", mu=31.15, sigma=20.83)
            sigma = pm.Gamma("sigma", mu=120.41, sigma=138.90)

            blocks = dataframe.set_index('block').groupby('block')
            for block, data in blocks:
                obs = pm.Gamma(f"obs_block{block}_parallel", mu=mu, sigma=sigma, observed=data['parallel'])
                obs = pm.Gamma(f"obs_block{block}_orthogonal", mu=mu, sigma=sigma, observed=data['orthogonal'])
        return model
    
    def get_model_1(self, dataframe):
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
        with pm.Model() as model:
            model.name = "Precision dependent on Synergy"
            # Assume that the squared deviations are drawn from a Gamma distribution with unknown parameters.
            # Make use of the divisibility of the gamma distribution: the sum of two gamma-distributed
            # variables is gamma-distributed again.
            scale = pm.Gamma("scale", alpha=1.0, beta=1.0)
            shape_ortho_1_3 = pm.Gamma("shape", alpha=1.0, beta=1.0)
            delta_shape_ortho_2 = pm.Gamma("dshape", alpha=1.0, beta=1.0)
            delta_shape_para_1_3 = pm.Gamma("dshape2", alpha=1.0, beta=1.0)

            # Describe blocks 1&3.
            for block in [1, 3]:
                data = dataframe[dataframe['block'] == block]
                # Blocks 1&3 orthogonal projections have low deviations.
                obs = pm.Gamma(f"obs_block{block}_orthogonal", alpha=shape_ortho_1_3, beta=scale,
                               observed=data['orthogonal'])
                # Parallel projections in blocks 1&3 have even higher deviations than in block 2.
                obs = pm.Gamma(f"obs_block{block}_parallel",
                               alpha=shape_ortho_1_3 + delta_shape_ortho_2 + delta_shape_para_1_3, beta=scale,
                               observed=data['parallel'])

            # Describe block 2.
            block2 = dataframe[dataframe['block'] == 2]
            # Block 2 parallel projections have the same average deviation as block 2 orthogonal projections.
            for projection in ['orthogonal', 'parallel']:
                # Block 2 orthogonal projections have a higher deviation on average than blocks 1&3.
                obs = pm.Gamma(f"obs_block2_{projection}", alpha=shape_ortho_1_3 + delta_shape_ortho_2, beta=scale,
                               observed=block2[projection])
        return model
    
    def get_model_2(self, dataframe):
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
        with pm.Model() as model:
            model.name = "Precision independent of Synergy"
            # Assume that the squared deviations are drawn from a Gamma distribution with unknown parameters.
            # Make use of the divisibility of the gamma distribution: the sum of two gamma-distributed
            # variables is gamma-distributed again.
            scale = pm.Gamma("scale", alpha=1.0, beta=1.0)
            shape_ortho = pm.Gamma("shape", alpha=1.0, beta=1.0)
            delta_shape_para_1_3 = pm.Gamma("dshape", alpha=1.0, beta=1.0)

            blocks = dataframe.set_index('block').groupby('block')
            for block, data in blocks:
                # All blocks' orthogonal projections have low deviations.
                obs = pm.Gamma(f"obs_block{block}_orthogonal", alpha=shape_ortho, beta=scale,
                               observed=data['orthogonal'])
                # Parallel projections in blocks 1&3 have higher deviations than in block 2.
                if block == 2:
                    obs = pm.Gamma(f"obs_block{block}_parallel", alpha=shape_ortho, beta=scale,
                                   observed=data['parallel'])
                else:
                    obs = pm.Gamma(f"obs_block{block}_parallel", alpha=shape_ortho + delta_shape_para_1_3, beta=scale,
                                   observed=data['parallel'])
        return model

    def get_model_3(self, dataframe):
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
        with pm.Model() as model:
            model.name = "Training Effect"
            # Assume that the squared deviations are drawn from a Gamma distribution with unknown parameters.
            # Make use of the divisibility of the gamma distribution: the sum of two gamma-distributed
            # variables is gamma-distributed again.
            scale = pm.Gamma("scale", alpha=1.0, beta=1.0)
            shape_ortho = pm.Gamma("shape", alpha=1.0, beta=1.0)
            delta_shape_para_3 = pm.Gamma("dshape", alpha=1.0, beta=1.0)
            delta_shape_para_1 = pm.Gamma("dshape2", alpha=1.0, beta=1.0)  # on top of dshape

            blocks = dataframe.set_index('block').groupby('block')
            for block, data in blocks:
                # All blocks' orthogonal projections have low deviations.
                obs = pm.Gamma(f"obs_block{block}_orthogonal", alpha=shape_ortho, beta=scale,
                               observed=data['orthogonal'])
                if block == 1:
                    obs = pm.Gamma(f"obs_block{block}_parallel",
                                   alpha=shape_ortho + delta_shape_para_1 + delta_shape_para_3, beta=scale,
                                   observed=data['parallel'])  # strong synergy
                elif block == 2:
                    # No synergy.
                    obs = pm.Gamma(f"obs_block{block}_parallel", alpha=shape_ortho, beta=scale,
                                   observed=data['parallel'])
                else:
                    obs = pm.Gamma(f"obs_block{block}_parallel", alpha=shape_ortho + delta_shape_para_3, beta=scale,
                                   observed=data['parallel'])  # weak synergy
        return model

    def get_model_4(self, dataframe):
        """ Compute marginal-log-likelihood of M4: Extraneous cognitive load / split-attention-effect.
        
        Constrained task is perceived more difficult by visual instructions. Strong synergy in simpler task.
        
        Prior believe: Likely for users with no video-game experience. It's also likely that participants don't figure
        out the control scheme of df1=df2=target/2, since only one df seems constrained which might get more attention.
        
        :type dataframe: pandas.DataFrame
        :rtype: float
        """
        with pm.Model() as model:
            model.name = "Split Attention Effect"
            # Assume that the squared deviations are drawn from a Gamma distribution with unknown parameters.
            # Make use of the divisibility of the gamma distribution: the sum of two gamma-distributed
            # variables is gamma-distributed again.
            scale = pm.Gamma("scale", alpha=1.0, beta=1.0)
            shape = pm.Gamma("shape", alpha=1.0, beta=1.0)
            delta_shape = pm.Gamma("dshape", alpha=1.0, beta=1.0)

            blocks = dataframe.set_index('block').groupby('block')
            for block, data in blocks:
                # All blocks' parallel projections have higher deviations than orthogonal projections in block 1&3.
                obs = pm.Gamma(f"obs_block{block}_parallel", alpha=shape + delta_shape, beta=scale,
                               observed=data['parallel'])
                # Orthogonal projections in blocks 2 have higher deviations than in block 1&3.
                if block == 2:
                    obs = pm.Gamma(f"obs_block{block}_orthogonal", alpha=shape + delta_shape, beta=scale,
                                   observed=data['orthogonal'])
                else:
                    obs = pm.Gamma(f"obs_block{block}_orthogonal", alpha=shape, beta=scale, observed=data['orthogonal'])
        return model

    def get_model_5(self, dataframe):
        """ Compute marginal-log-likelihood of M5: Synergy is needed to perform a task.
        
        Main effect of projection (parallel > orthogonal).
        Strong synergies in all the tasks. This model implies a bad performance for the additional goal in the
        constrained task.
        
        Prior believe: Unlikely.
        
        :type dataframe: pandas.DataFrame
        :rtype: float
        """
        with pm.Model() as model:
            model.name = "Main Effect Projection"
            # Assume that the squared deviations are drawn from a Gamma distribution with unknown parameters.
            # Make use of the divisibility of the gamma distribution: the sum of two gamma-distributed
            # variables is gamma-distributed again.
            scale = pm.Gamma("scale", alpha=1.0, beta=1.0)
            shape_ortho = pm.Gamma("shape", alpha=1.0, beta=1.0)
            delta_shape_para = pm.Gamma("dshape", alpha=1.0, beta=1.0)
        
            # Describe blocks.
            blocks = dataframe.set_index('block').groupby('block')
            for block, data in blocks:
                # All blocks' orthogonal projections have low deviations.
                obs = pm.Gamma(f"obs_block{block}_orthogonal", alpha=shape_ortho, beta=scale,
                               observed=data['orthogonal'])
                # All blocks' parallel projections have higher deviations than the orthogonal projections.
                obs = pm.Gamma(f"obs_block{block}_parallel", alpha=shape_ortho + delta_shape_para, beta=scale,
                               observed=data['parallel'])
        return model

    def get_model_6(self, dataframe):
        """ Compute marginal-log-likelihood of M6:
        
        Main effect of treatment (block 2 > block 1&3).
        No synergy effects can be observed for this particular bi-manual task. The constrained task (block 2) appears
        too difficult to coordinate and results in heightened deviations in all directions compared to the
        unconstrained task (blocks 1&3).
        
        Prior believe: Unlikely
        
        :type dataframe: pandas.DataFrame
        :rtype: float
        """
        with pm.Model() as model:
            model.name = "Main Effect Block"
            # Assume that the squared deviations are drawn from a Gamma distribution with unknown parameters.
            # Both parameters have to be positive, so draw them from a Gamma prior, too.
            # TODO: Perhaps use the alternative parametrization with mu, sigma instead, once we have the pilot data.
            scale = pm.Gamma("scale", alpha=1.0, beta=1.0)
            base_shape = pm.Gamma("shape", alpha=1.0, beta=1.0)
            delta_shape = pm.Gamma("dshape", alpha=1.0, beta=1.0)
        
            blocks = dataframe.set_index('block').groupby('block')
            for block, data in blocks:
                if block == 2:
                    shape = base_shape + delta_shape
                else:
                    shape = base_shape
                obs = pm.Gamma(f"obs_block{block}_parallel", alpha=shape, beta=scale, observed=data['parallel'])
                obs = pm.Gamma(f"obs_block{block}_orthogonal", alpha=shape, beta=scale, observed=data['orthogonal'])
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
            print(f"Model {i}: '{model.name}'...", end="")
            model = func(data)
            models.append(model)
            print("done.")
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
            trace = pm.sample(self.draws, tune=self.tune)
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
        models = self.get_models(df)
        # First, compute all marginal log-likelihoods,
        model_posterior = np.array([self.get_marginal_likelihood(model) for model in models])
        # then, exponentiate and normalize to turn into a probability distribution.
        model_posterior = np.exp(model_posterior-np.logaddexp.reduce(model_posterior))
        # Save posterior for this participant.
        self.posteriors.loc[user] = model_posterior
        return {i: mp for i, mp in enumerate(model_posterior)}

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
