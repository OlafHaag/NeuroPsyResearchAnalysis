# From https://junpenglao.xyz/Blogs/posts/2017-11-22-Marginal_likelihood_in_PyMC3.html
# Based on https://github.com/quentingronau/bridgesampling/blob/master/R/bridge_sampler_normal.R

import pymc3 as pm
from pymc3.model import modelcontext
from scipy.linalg import cholesky as chol
import warnings
import numpy as np
import scipy.stats as st


def marginal_llk(mtrace, model=None, logp=None, maxiter=1000):
    """The Bridge Sampling Estimator of the Marginal Likelihood.

    Parameters
    ----------
    mtrace : MultiTrace, result of MCMC run
    model : PyMC Model
        Optional model. Default None, taken from context.
    logp : Model Log-probability function, read from the model by default
    maxiter : Maximum number of iterations

    Returns
    -------
    marg_llk : Estimated Marginal log-Likelihood.
    """
    r0, tol1, tol2 = 0.5, 1e-10, 1e-4

    model = modelcontext(model)
    with model:
        if logp is None:
            logp = model.logp_array
            
        # free_RVs might be autotransformed. 
        # if that happens, there will be a model.deterministics entry with the first part of the name that needs to be used 
        # instead of the autotransformed name below in stats.ess
        # so we need to replace that variable with the corresponding one from the deterministics
        vars = model.free_RVs
        det_names=[d.name for d in model.deterministics]
        det_names.sort(key=lambda s:-len(s)) # sort descending by length
        
        def recover_var_name(name_autotransformed):
            for dname in det_names:
                if dname==name_autotransformed[:len(dname)]:
                    return dname
            return name_autotransformed

        # Split the samples into two parts  
        # Use the first 50% for fiting the proposal distribution and the second 50% 
        # in the iterative scheme.
        len_trace = len(mtrace)
        nchain = mtrace.nchains
        
        N1_ = len_trace // 2
        N1 = N1_*nchain
        N2 = len_trace*nchain - N1

        neff_list = dict() # effective sample size

        arraysz = model.bijection.ordering.size
        samples_4_fit = np.zeros((arraysz, N1))
        samples_4_iter = np.zeros((arraysz, N2))
        # matrix with already transformed samples
        for var in vars:
            varmap = model.bijection.ordering.by_name[var.name]
            # for fitting the proposal
            x = mtrace[:N1_][var.name]
            samples_4_fit[varmap.slc, :] = x.reshape((x.shape[0], np.prod(x.shape[1:], dtype=int))).T
            # for the iterative scheme
            x2 = mtrace[N1_:][var.name]
            samples_4_iter[varmap.slc, :] = x2.reshape((x2.shape[0], np.prod(x2.shape[1:], dtype=int))).T
            # effective sample size of samples_4_iter, scalar
            orig_name=recover_var_name(var.name)
            neff_list.update(pm.stats.ess(mtrace[N1_:], var_names=[orig_name]))

        # median effective sample size (scalar)
        # ToDo: Use arviz summary to get median effective sample size?
        neff = np.median(np.concatenate([x.values.reshape((1,)) if x.shape==() else x for x in neff_list.values()]))
        
        # get mean & covariance matrix and generate samples from proposal
        m = np.mean(samples_4_fit, axis=1)
        V = np.cov(samples_4_fit)
        L = chol(V, lower=True)

        # Draw N2 samples from the proposal distribution
        gen_samples = m[:, None] + np.dot(L, st.norm.rvs(0, 1, size=samples_4_iter.shape))

        # Evaluate proposal distribution for posterior & generated samples
        q12 = st.multivariate_normal.logpdf(samples_4_iter.T, m, V)
        q22 = st.multivariate_normal.logpdf(gen_samples.T, m, V)

        # Evaluate unnormalized posterior for posterior & generated samples
        q11 = np.asarray([logp(point) for point in samples_4_iter.T])
        q21 = np.asarray([logp(point) for point in gen_samples.T])

        # Iterative scheme as proposed in Meng and Wong (1996) to estimate
        # the marginal likelihood
        def iterative_scheme(q11, q12, q21, q22, r0, neff, tol, maxiter, criterion):
            l1 = q11 - q12
            l2 = q21 - q22
            lstar = np.median(l1) # To increase numerical stability, 
                                # subtracting the median of l1 from l1 & l2 later
            s1 = neff/(neff + N2)
            s2 = N2/(neff + N2)
            r = r0
            r_vals = [r]
            logml = np.log(r) + lstar
            criterion_val = 1 + tol

            i = 0
            while (i <= maxiter) & (criterion_val > tol):
                rold = r
                logmlold = logml
                numi = np.exp(l2 - lstar)/(s1 * np.exp(l2 - lstar) + s2 * r)
                deni = 1/(s1 * np.exp(l1 - lstar) + s2 * r)
                if np.sum(~np.isfinite(numi))+np.sum(~np.isfinite(deni)) > 0:
                    warnings.warn("""Infinite value in iterative scheme, returning NaN. 
                    Try rerunning with more samples.""")
                r = (N1/N2) * np.sum(numi)/np.sum(deni)
                r_vals.append(r)
                logml = np.log(r) + lstar
                i += 1
                if criterion=='r':
                    criterion_val = np.abs((r - rold)/r)
                elif criterion=='logml':
                    criterion_val = np.abs((logml - logmlold)/logml)

            if i >= maxiter:
                return dict(logml = np.NaN, niter = i, r_vals = np.asarray(r_vals))
            else:
                return dict(logml = logml, niter = i)

        # Run iterative scheme:
        tmp = iterative_scheme(q11, q12, q21, q22, r0, neff, tol1, maxiter, 'r')
        if ~np.isfinite(tmp['logml']):
            warnings.warn("""logml could not be estimated within maxiter, rerunning with 
                        adjusted starting value. Estimate might be more variable than usual.""")
            # use geometric mean as starting value
            r0_2 = np.sqrt(tmp['r_vals'][-2]*tmp['r_vals'][-1])
            tmp = iterative_scheme(q11, q12, q21, q22, r0_2, neff, tol2, maxiter, 'logml')

    return dict(logml = tmp['logml'], niter = tmp['niter'], method = "normal", 
                q11 = q11, q12 = q12, q21 = q21, q22 = q22)
