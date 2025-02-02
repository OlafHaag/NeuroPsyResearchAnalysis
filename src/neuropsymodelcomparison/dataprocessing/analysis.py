""" This file originated from the online analysis project at:
https://github.com/OlafHaag/UCM-WebApp
"""

import itertools

import pandas as pd
import pingouin as pg
import numpy as np
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope


def preprocess_data(users, blocks, trials):
    """ Clean data.
    
    :param users: Data from users table
    :type users: pandas.DataFrame
    :param blocks: Data from circletask_blocks table.
    :type blocks: pandas.DataFrame
    :param trials: Data from circletask_trials table.
    :type trials: pandas.DataFrame
    :returns: Joined and recoded DataFrame. Number of erroneous blocks. Number of sessions removed as a consequence.
              Number of removed trials.
    :rtype: tuple[pandas.DataFrame, int, int, int]
    """
    blocks, n_errors, invalid_sessions = remove_erroneous_blocks(blocks)
    # Merge to 1 table.
    df = join_data(users, blocks, trials)
    # Remove invalid trials.
    cleaned, n_trials_removed = get_valid_trials(df)
    return cleaned, n_errors, len(invalid_sessions), n_trials_removed


def remove_erroneous_blocks(blocks, delta_time=2.0, n_blocks=3):
    """ Remove sessions with erroneous data due to a NeuroPsy Research App malfunction.
    The error causes block data to be duplicated and the values for df1 & df2 multiplied again by 100.
    The duplicated blocks are identified by comparing their time stamps to the previous block (less than 2 seconds
    difference). If the error caused the session to end early, the whole session is removed.
    
    NeuroPsyResearchApp issue #1.
    
    :param pandas.DataFrame blocks: Data about blocks.
    :param float delta_time: Threshold in seconds for which a consecutive block in a session is considered invalid
                             if it was completed within this period after the previous. Default is 2.0 seconds.
    :param int n_blocks: Required number of blocks per session. If a session doesn't have this many blocks,
                         it gets removed.
    :returns: Cleaned block data. Number of errors found. List of sessions that were removed as a consequence.
    :rtype: tuple[pandas.DataFrame, int, list]
    """
    # Identify duplicated blocks. Consecutive time stamps are usually less than 2 seconds apart.
    mask = blocks.groupby(['session_uid'])['time'].diff() < delta_time
    try:
        n_errors = mask.value_counts()[True]
    except KeyError:
        n_errors = 0
    blocks = blocks.loc[~mask, :]
    # Now, after removal of erroneous data a session might not have all 3 blocks we expect. Exclude whole session.
    invalid_sessions = blocks['session_uid'].value_counts() != n_blocks
    invalid_sessions = invalid_sessions.loc[invalid_sessions].index.to_list()
    blocks = blocks.loc[~blocks['session_uid'].isin(invalid_sessions), :]
    return blocks, n_errors, invalid_sessions


def join_data(users, blocks, trials):
    """ Take data from different database tables and join them to a single DataFrame. Some variables are renamed and
    recoded in the process, some are dropped.
    
    :param users: Data from users table
    :type users: pandas.DataFrame
    :param blocks: Data from circletask_blocks table.
    :type blocks: pandas.DataFrame
    :param trials: Data from circletask_trials table.
    :type trials: pandas.DataFrame
    :return: Joined and recoded DataFrame.
    :rtype: pandas.DataFrame
    """
    # Use users' index instead of id for obfuscation and shorter display.
    users_inv_map = pd.Series(users.index, index=users.id)
    # Remove trials that don't belong to any block. Those have been excluded.
    trials = trials.loc[trials['block_id'].isin(blocks.index), :]
    # Start a new table for trials and augment with data from other tables.
    df = pd.DataFrame(index=trials.index)
    df['user'] = trials.user_id.map(users_inv_map).astype('category')
    df['session'] = trials['block_id'].map(blocks['nth_session']).astype('category')
    # Map whole sessions to the constraint in the treatment block as a condition for easier grouping during analysis.
    df['condition'] = trials['block_id'].map(blocks[['session_uid', 'treatment']].replace(
        {'treatment': {'': np.nan}}).groupby('session_uid')['treatment'].ffill().bfill()).astype('category')
    df['block'] = trials['block_id'].map(blocks['nth_block']).astype('category')
    # Add pre and post labels to trials for each block. Name it task instead of treatment.
    # Theoretically, one could have changed number of blocks and order of treatment, but we assume default order here.
    df['task'] = trials['block_id'].map(blocks['treatment'].replace('', np.nan).where(~blocks['treatment'].isna(),
                                                                                      blocks['nth_block'].map(
                                                                                          {1: 'pre',
                                                                                           3: 'post'
                                                                                           })
                                                                                      )
                                        ).astype('category')
    #df['task'] = trials['block_id'].map(blocks['treatment'].replace(to_replace={r'\w+': 1, r'^\s*$': 0}, regex=True)
    #                                   ).astype('category')
    
    df = pd.concat((df, trials), axis='columns')
    # Add columns for easier filtering.
    df['grab_diff'] = (df['df2_grab'] - df['df1_grab']).abs()
    df['duration_diff'] = (df['df2_duration'] - df['df1_duration']).abs()
    # Exclude columns.
    df.drop(columns=['user_id'], inplace=True)
    return df


def get_valid_trials(dataframe):
    """ Remove trials where sliders where not grabbed concurrently or grabbed at all.
    
    :param dataframe: Trial data.
    :type dataframe: pandas.DataFrame
    :returns: Filtered trials. Number of removed trials.
    :rtype: tuple[pandas.DataFrame, int]
    """
    # Remove trials with missing values. This means at least one slider wasn't grabbed.
    df = dataframe.dropna(axis='index', how='any')
    # Remove trials where sliders where not grabbed concurrently.
    mask = ~((df['df1_release'] <= df['df2_grab']) | (df['df2_release'] <= df['df1_grab']))
    df = df.loc[mask, :]
    n_removed = len(dataframe) - len(df)
    return df, n_removed
    
    
def get_outlyingness(data, contamination=0.1):
    """ Outlier detection from covariance estimation in a Gaussian distributed dataset.
    
    :param data: Data in which to detect outliers. Take care that n_samples > n_features ** 2 .
    :type data: pandas.DataFrame
    :param contamination: The amount of contamination of the data set, i.e. the proportion of outliers in the data set.
    Range is (0, 0.5).
    :type contamination: float
    :returns: Decision on each row if it's an outlier. And contour array for drawing ellipse in graph.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    robust_cov = EllipticEnvelope(support_fraction=1., contamination=contamination)
    outlyingness = robust_cov.fit_predict(data)
    decision = (outlyingness-1).astype(bool)
    
    # Visualisation.
    xx, yy = np.meshgrid(np.linspace(0, 100, 101),
                         np.linspace(0, 100, 101))
    z = robust_cov.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    
    return decision, z
    #ToDo: remove blocks/sessions with sum mean way off.
    #ToDo: remove sessions with less than 10 trials in any block.


def get_performance_data(dataframe):
    """[summary]

    :param dataframe: [description]
    :type dataframe: [type]
    """
    dataframe.groupby(['user', 'block', 'task'])[['df1', 'df2']].mean().dropna().sort_index(level=['user','block'])


def get_pca_data(dataframe):
    """ Conduct Principal Component Analysis on 2D dataset.
    
    :param dataframe: Data holding 'df1' and 'df2' values as columns.
    :type dataframe: pandas.DataFrame
    :return: Explained variance, components and means.
    :rtype: pandas.DataFrame
    """
    # We don't reduce dimensionality, but overlay the 2 principal components in 2D.
    pca = PCA(n_components=2)
    
    x = dataframe[['df1', 'df2']].values
    try:
        # df1 and df2 have the same scale. No need to standardize. Standardizing might actually distort PCA here.
        pca.fit(x)
    except ValueError:
        # Return empty.
        df = pd.DataFrame(columns=['var_expl', 'var_expl_ratio', 'x', 'y', 'meanx', 'meany'])
    else:
        df = pd.DataFrame({'var_expl': pca.explained_variance_.T,
                           'var_expl_ratio': pca.explained_variance_ratio_.T * 100,  # In percent
                           'x': pca.components_[:, 0],
                           'y': pca.components_[:, 1],
                           'meanx': pca.mean_[0],
                           'meany': pca.mean_[1],
                           },
                          index=[1, 2]  # For designating principal components.
                          )
    df.index.rename('PC', inplace=True)
    return df


def get_pca_vectors(dataframe):
    """ Get principal components as vectors. Vectors can then be used to annotate graphs.
    
    :param dataframe: Tabular PCA data.
    :type dataframe: pandas.DataFrame
    :return: Principal components as vector pairs in input space with mean as origin first and offset second.
    :rtype: list
    """
    # Use the "components" to define the direction of the vectors,
    # and the "explained variance" to define the squared-length of the vectors.
    directions = dataframe[['x', 'y']] * np.sqrt(dataframe[['var_expl']].values) * 3
    # Move the directions by the mean, so we get vectors pointing to the start and vectors pointing to the destination.
    vector2 = directions + dataframe[['meanx', 'meany']].values
    vectors = list(zip(dataframe[['meanx', 'meany']].values, vector2.values))
    return vectors


def get_pca_vectors_by(dataframe, by=None):
    """ Get principal components for each group as vectors. Vectors can then be used to annotate graphs.

    :param dataframe: Data holding 'df1' and 'df2' values as columns.
    :type dataframe: pandas.DataFrame
    :param by: Column to group data by and return 2 vectors for each group.
    :type by: str|list
    :return: list of principal components as vector pairs in input space with mean as origin first and offset second.
    :rtype: list
    """
    vector_pairs = list()
    if by is None:
        pca_df = get_pca_data(dataframe)
        v = get_pca_vectors(pca_df)
        vector_pairs.append(v)
    else:
        grouped = dataframe.groupby(by, observed=True)  # With categorical groupers we want only non-empty groups.
        for group, data in grouped:
            pca_df = get_pca_data(data)
            v = get_pca_vectors(pca_df)
            vector_pairs.append(v)
            # ToDo: Augment by groupby criteria.
            
    return vector_pairs


def get_interior_angle(vec0, vec1):
    """ Get the smaller angle between vec0 and vec1 in degrees.
    
    :param vec0: Vector 0
    :type vec0: numpy.ndarray
    :param vec1: Vector 1
    :type vec1: numpy.ndarray
    :return: Interior angle between vector0 and vector1 in degrees.
    :rtype: float
    """
    angle = np.math.atan2(np.linalg.det([vec0, vec1]), np.dot(vec0, vec1))
    degrees = abs(np.degrees(angle))
    # Min and max should be between 0° an 90°.
    degrees = min(degrees, 180.0 - degrees)
    return degrees


def get_ucm_vec(p0=None, p1=None):
    """ Returns 2D unit vector in direction of uncontrolled manifold. """
    if p0 is None:
        p0 = np.array([25, 100])
    if p1 is None:
        p1 = np.array([100, 25])
    parallel = p1 - p0
    parallel = parallel / np.linalg.norm(parallel)  # Normalize.
    return parallel


def get_orthogonal_vec2d(vec):
    """ Get a vector that is orthogonal to vec and has same length.
    
    :param vec: 2D Vector
    :return: 2D Vector orthogonal to vec.
    :rtype: numpy.ndarray
    """
    ortho = np.array([-vec[1], vec[0]])
    return ortho


def get_pc_ucm_angles(dataframe, vec_ucm):
    """ Computes the interior angles between pca vectors and ucm parallel/orthogonal vectors.
    
    :param dataframe: PCA data.
    :type dataframe: pandas.DataFrame
    :param vec_ucm: Vector parallel to UCM.
    :type vec_ucm: numpy.ndarray
    :return: Each angle between principal components and UCM parallel and orthogonal vector.
    :rtype: pandas.DataFrame
    """
    df_angles = dataframe[['x', 'y']].transform(lambda x: (a:=get_interior_angle(vec_ucm, x), 90.0 - a),
                                                axis='columns').rename(columns={'x': 'parallel', 'y': 'orthogonal'})
    df_angles = pd.concat((dataframe[['task', 'PC']], df_angles), axis='columns')
    return df_angles


def get_projections(points, vec_ucm):
    """ Returns coefficients a and b in x = a*vec_ucm + b*vec_ortho with x being the difference of a data point and
    the mean.
    Projection is computed using a transformation matrix with ucm parallel and orthogonal vectors as basis.
    
    :param points: Data of 2D points.
    :type points: pandas.Dataframe
    :param vec_ucm: Unit vector parallel to uncontrolled manifold.
    :type vec_ucm: numpy.ndarray
    :return: Array with projected lengths onto vector parallel to UCM as 'a', onto vector orthogonal to UCM as 'b'.
    :rtype: pandas.Dataframe
    """
    # Get the vector orthogonal to the UCM.
    vec_ortho = get_orthogonal_vec2d(vec_ucm)
    # Build a transformation matrix with vec_ucm and vec_ortho as new basis vectors.
    A = np.vstack((vec_ucm, vec_ortho)).T  # A is not an orthogonal projection matrix (A=A.T), but this works.
    # Centralize the data. Analogous to calculating across trials deviation from average for each time step.
    diffs = points - points.mean()
    # For computational efficiency we shortcut the projection calculation with matrix multiplication.
    # The actual math behind it:
    #   coeffs = vec_ucm.T@diff/np.sqrt(vec_ucm.T@vec_ucm), vec_ortho.T@diff/np.sqrt(vec_ortho.T@vec_ortho)
    # Biased variance (normalized by (n-1)) of projection onto UCM vector:
    #   var_ucm = vec_ucm.T@np.cov(diffs, bias=True, rowvar=False)@vec_ucm/(vec_ucm.T@vec_ucm)  # Rayleigh fraction.
    coeffs = diffs@A
    coeffs.columns = ['parallel', 'orthogonal']
    return coeffs


def get_synergy_indices(variances, n=2, d=1):
    """
    n: Number of degrees of freedom. In our case 2.
    
    d: Dimensionality of performance variable. In our case a scalar (1).
    
    Vucm = 1/N * 1/(n - d) * sum(ProjUCM**2)
    
    Vort = 1/N * 1/(d) * sum(ProjORT**2)
    
    Vtotal = 1/n * (d * Vort + (n-d) * Vucm)  # Anull the weights on Vucm and Vort for the sum.
    
    dV = (Vucm - Vort) / Vtotal
    dV = n*(Vucm - Vort) / ((n - d)*Vucm + d*Vort)
    
    Zhang (2008) without weighting Vucm, Vort and Vtotal first:
    dV = n * (Vucm/(n - d) - Vort/d) / (Vucm + Vort)
    
    dVz = 0.5*ln((n / d + dV) / (n / ((n - d) - dV))
    dVz = 0.5*ln((2 + dV) / (2 - dV))
    Reference: https://www.frontiersin.org/articles/10.3389/fnagi.2019.00032/full#supplementary-material
    
    :param variances: Unweighted variances of parallel and orthogonal projections to the UCM.
    :type variances: pandas.DataFrame
    :param n: Number of degrees of freedom. Defaults to 2.
    :type: int
    :param d: Dimensionality of performance variable. Defaults to 1.
    :type d: int
    :returns: Synergy index, Fisher's z-transformed synergy index.
    :rtype: pandas.DataFrame
    """
    try:
        dV = n * (variances['parallel']/(n-d) - variances['orthogonal']/d) \
             / variances[['parallel', 'orthogonal']].sum(axis='columns')
    except KeyError:
        synergy_indices = pd.DataFrame(columns=["dV", "dVz"])
    else:
        dVz = 0.5 * np.log((n/d + dV)/(n/(n-d) - dV))
        synergy_indices = pd.DataFrame({"dV": dV, "dVz": dVz})
    return synergy_indices


def get_synergy_idx_bounds(n=2, d=1):
    """ Get lower and upper bounds of the synergy index.
    
     dV = n * (Vucm/(n - d) - Vort/d) / (Vucm + Vort)
     
     If all variance lies within the UCM, then Vort=0 and it follows for the upper bound: dV = n/(n-d)
     
     If all variance lies within Vort, then Vucm=0 and it follows for the lower bound: dV = -n/d
    
    :param n: Number of degrees of freedom.
    :type: int
    :param d: Dimensionality of performance variable.
    :type d: int
    :returns: Lower and upper bounds of synergy index.
    :rtype: tuple
    """
    dV_lower = -n/d
    dV_upper = n/(n-d)
    return dV_lower, dV_upper
    
    
def get_mean(dataframe, column, by=None):
    """ Return mean values of column x (optionally grouped)
    
    :param dataframe: Data
    :type dataframe: pandas.Dataframe
    :param column: Column name
    :type column: str
    :param by: Column names by which to group.
    :type by: str|list
    :return: mean value, optionally for each group.
    :rtype: numpy.float64|pandas.Series
    """
    if by is None:
        means = dataframe[column].mean()
    else:
        means = dataframe.groupby(by, observed=True)[column].mean()
    return means


def get_descriptive_stats(data, by=None):
    """ Return mean and variance statistics for data.
    
    :param data: numerical data.
    :type data: pandas.Dataframe
    :param by: groupby column name(s)
    :type by: str|List
    :return: Dataframe with columns mean, var, count and column names of data as rows.
    :rtype: pandas.Dataframe
    """
    # There's a bug in pandas 1.0.4 where you can't use custom numpy functions in agg anymore (ValueError).
    # Note that the variance of projections is usually divided by (n-d) for Vucm and d for Vort. Both are 1 in our case.
    # Pandas default var returns unbiased population variance /(n-1). Doesn't make a difference for synergy indices.
    f_var = lambda series: series.var(ddof=0)
    f_var.__name__ = 'variance'  # Column name gets function name.
    # When there're no data, return empty DataFrame with columns.
    if data.empty:
        if by:
            data.set_index(by, drop=True, inplace=True)
        col_idx = pd.MultiIndex.from_product([data.columns, ['mean', f_var.__name__]])
        stats = pd.DataFrame(None, index=data.index, columns=col_idx)
        stats['count'] = None
        return stats
    
    if not by:
        stats = data.agg(['mean', f_var, 'count']).T
        stats['count'] = stats['count'].astype(int)
    else:
        grouped = data.groupby(by, observed=True)
        stats = grouped.agg(['mean', f_var])
        stats['count'] = grouped.size()
        stats.dropna(inplace=True)
    return stats


def get_statistics(dataframe):
    """ Calculate descriptive statistics including synergy indices for key values of the anaylsis.

    :param dataframe: Data from joined table on trials with projections.
    :type dataframe: pandas.DataFrame
    :return: Descriptive statistics and synergy indices.
    :rtype: pandas.DataFrame
    """
    groupers = ['user', 'session', 'condition', 'block_id', 'block', 'task']
    try:
        dataframe[groupers] = dataframe[groupers].astype('category')
    except (KeyError, ValueError):
        df_stats = get_descriptive_stats(pd.DataFrame(columns=dataframe.columns))
        cov = pd.DataFrame(columns=[('df1,df2 covariance', '')])
    else:
        df_stats = get_descriptive_stats(dataframe[groupers + ['df1', 'df2', 'sum', 'parallel', 'orthogonal']],
                                                  by=groupers).drop(columns=[('parallel', 'mean'),  # Always equal 0.
                                                                             ('orthogonal', 'mean')])
        # Get statistic characteristics of absolute lengths of projections.
        length = dataframe.groupby(groupers, observed=True)[['parallel', 'orthogonal']].agg(lambda x: x.abs().mean())
        length.columns = pd.MultiIndex.from_product([length.columns, ['absolute average']])
        # Get covariance between degrees of freedom.
        cov = dataframe.groupby(groupers, observed=True)[['df1', 'df2']].apply(lambda x: np.cov(x.T, ddof=0)[0, 1])
        try:
            cov = cov.to_frame(('df1,df2 covariance', ''))  # MultiIndex.
        except AttributeError:  # In case cov is an empty Dataframe.
            cov = pd.DataFrame(columns=pd.MultiIndex.from_tuples([('df1,df2 covariance', '')]))

    # Get synergy indices based on projection variances we just calculated.
    df_synergies = get_synergy_indices(df_stats[['parallel', 'orthogonal']].xs('variance', level=1, axis='columns'))
    # Before we merge dataframes, give this one a Multiindex, too.
    df_synergies.columns = pd.MultiIndex.from_product([df_synergies.columns, ['']])
    # Join the 3 statistics to be displayed in a single table.
    df = pd.concat((df_stats, cov, length, df_synergies), axis='columns')
    # Sort the columns manually.
    df = df.sort_index(axis='columns', level=0)[['count', 'df1', 'df2', 'df1,df2 covariance', 'sum', 
                                                 'parallel', 'orthogonal', 'dV', 'dVz']]
    return df


def wilcoxon_rank_test(data):
    w, p = wilcoxon(data['parallel'], data['orthogonal'], alternative='greater')
    return p < 0.05, w, p


def wide_to_long(df, stubs, suffixes, j):
    """ Transforms a dataframe to long format, where the stubs are melted into a single column with name j and suffixes
    into value columns. Filters for all columns that are a stubs+suffixes combination.
    Keeps 'user', 'task' as id_vars. When an error is encountered an emtpy dataframe is returned.
    
    :param df: Data in wide/mixed format.
    :type df: pandas.DataFrame
    :param stubs: First part of a column name. These names will be the values of the new column j.
    :type stubs: list[str]
    :param suffixes: Second part of a column name. These will be the new columns holding the respective values.
    :type suffixes: str|list[str]
    :param j: Name for new column containing stubs.
    :type j: str
    
    :return: Filtered Dataframe in long format.
    :rtype: pandas.Dataframe
    """
    if isinstance(suffixes, str):
        suffixes = [suffixes]
    # We want all stubs+suffix combinations as columns.
    val_cols = [" ".join(x) for x in itertools.product(stubs, suffixes)]
    try:
        # Filter for data we want to plot.
        df = df[['user', 'condition', 'block', 'task', *val_cols]]
        # Reverse stub and suffix for long format. We want the measurements as columns, not the categories.
        df.columns = [" ".join(x.split(" ")[::-1]) for x in df.columns]
        long_df = pd.wide_to_long(df=df, stubnames=suffixes, i=['user', 'condition', 'block', 'task'],
                                  j=j, sep=" ", suffix=f'(!?{"|".join(stubs)})')
        long_df.reset_index(inplace=True)
    except (KeyError, ValueError):
        long_df = pd.DataFrame(columns=['user', 'condition', 'block', 'task', j, *suffixes])
    long_df[['user', 'condition', 'block', 'task']] = long_df[['user', 'condition', 'block', 'task']].astype('category')
    return long_df


def normality_test(df, columns, multivariate=False):
    """ Tests whether there is considerable deviation from a normal distribution.
    If no deviation could be detected, we don't know much about the distribution.
    Independent normality tests use the Shapiro-Wilk method. Multivariate tests use the Henze-Zirkler multivariate
    normality test.
    
    :param df: Aggregated data containing Fisher-z-transformed synergy index.
    :type df: pandas.DataFrame
    :param columns: Which columns to test for normality deviation.
    :type columns: list[str]
    :param multivariate: Do multivariate normality testing?
    :type multivariate: bool
    :return: Normality test results.
    :rtype: pandas.DataFrame
    """
    if multivariate:
        # Multivariate testing.
        is_normal, p = df.groupby(['user', 'block'], observed=True)[columns].apply(pg.multivariate_normality)
        res = df.groupby(['user', 'block'], observed=True)[['df1', 'df2']].apply(pg.multivariate_normality)\
            .apply(pd.Series)\
            .rename(columns={0: 'normal', 1: 'p'})
    else:
        # We would want to minimize type II error rate, risk of not rejecting the null when it's false.
        # Shapiro-Wilk tests.
        res = df.groupby(['user', 'block'], observed=True)[columns].apply(pg.normality).unstack(level=2)  
    return res


def mixed_anova_synergy_index_z(dataframe):
    """ 3 x (3) Two-way split-plot ANOVA with between-factor condition and within-factor block.
    
    :param dataframe: Aggregated data containing Fisher-z-transformed synergy index.
    :type dataframe: pandas.DataFrame
    :return: mixed-design ANOVA results.
    :rtype: pandas.DataFrame
    """
    if dataframe['condition'].nunique() <= 1:
        raise ValueError("ERROR: Between factor has insufficient number of levels.")
        #ToDo: If there's only 1 condition, run ANOVA with one within factor instead.
    if dataframe['block'].nunique() <= 1:
        raise ValueError("ERROR: Between factor has insufficient number of levels.")
        #ToDo: If there's only 1 block, run ANOVA with one between factor instead.
    aov = pg.mixed_anova(data=dataframe, dv='dVz', within='block', subject='user', between='condition', correction=True)
    return aov


def posthoc_ttests(dataframe, var_='dVz'):
    """ Pairwise posthoc t-tests on a variable in a mixed design. Between factor is 'condition', within factor is
    'block'.
    
    :param dataframe: Aggregated data containing Fisher-z-transformed synergy index in long format.
    :type dataframe: pandas.DataFrame
    :param var_: The variable which to test. One of the column names in dataframe.
    :type var_: str
    :return: Pairwise T-tests results.
    :rtype: pandas.DataFrame
    """
    posthocs = pg.pairwise_ttests(data=dataframe, dv=var_, within='block', subject='user', between='condition',
                                  alpha=0.05, within_first=False,
                                  padjust='fdr_by', marginal=True, return_desc=True, tail='one-sided', parametric=True)
    return posthocs
