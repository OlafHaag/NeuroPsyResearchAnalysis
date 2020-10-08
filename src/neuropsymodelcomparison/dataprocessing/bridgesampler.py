# From https://junpenglao.xyz/Blogs/posts/2017-11-22-Marginal_likelihood_in_PyMC3.html
# Based on https://github.com/quentingronau/bridgesampling/blob/master/R/bridge_sampler_normal.R
# unittests and adaptation to pymc3 version 

import pymc3
from pymc3.model import modelcontext
from scipy import dot
from scipy.linalg import cholesky as chol
import warnings
import numpy as np
import scipy.stats as st
import scipy.special as spf
import theano.tensor as tt

import unittest


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
        samples_4_fit[varmap.slc, :] = x.reshape((x.shape[0], 
                                                  np.prod(x.shape[1:], dtype=int))).T
        # for the iterative scheme
        x2 = mtrace[N1_:][var.name]
        samples_4_iter[varmap.slc, :] = x2.reshape((x2.shape[0], 
                                                    np.prod(x2.shape[1:], dtype=int))).T
        # effective sample size of samples_4_iter, scalar
        orig_name=recover_var_name(var.name)
        neff_list.update(pymc3.stats.ess(mtrace[N1_:],var_names=[orig_name]))

    # median effective sample size (scalar)
    neff = np.median(list(neff_list.values()))
    
    # get mean & covariance matrix and generate samples from proposal
    m = np.mean(samples_4_fit, axis=1)
    V = np.cov(samples_4_fit)
    L = chol(V, lower=True)

    # Draw N2 samples from the proposal distribution
    gen_samples = m[:, None] + np.dot(L, st.norm.rvs(0, 1, 
                                         size=samples_4_iter.shape))

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


class TestBridgeSampler(unittest.TestCase):
    
    def test_bernoulli_process(self):
        """Testing the Bridge Sampler with a Beta-Bernoulli-Process model"""
        
        # prior parameters
        alpha=np.random.gamma(1.0,2.0)
        beta=np.random.gamma(1.0,2.0)
        n=100
        
        draws=10000
        tune=1000
        
        print("Testing with alpha=",alpha,"and beta=",beta)
        
        # random data
        p0=np.random.random()
        expected_error=np.sqrt(p0*(1-p0)/n) # reasonable approximation
        
        observations=(np.random.random(n)<=p0).astype("int")
        
        with pymc3.Model() as BernoulliBeta:
    
            theta=pymc3.Beta('pspike',alpha=alpha,beta=beta)
            obs=pymc3.Categorical('obs',p=tt.stack([theta,1.0-theta]),observed=observations)
            trace=pymc3.sample(draws=draws,tune=tune)

        # calculate exact marginal likelihood
        n=len(observations)
        k=sum(observations)
        print(n,k)
        exact_log_marg_ll=spf.betaln(alpha+k,beta+(n-k))-spf.betaln(alpha,beta)
    
        # estimate with bridge sampling
        logml_dict=marginal_llk(trace, model=BernoulliBeta, maxiter=10000)
        expected_p=1.0-trace["pspike"].mean()
        
        # should be true in 95% of the runs
        self.assertTrue(np.abs(expected_p-p0)<2*expected_error,
                        msg="Estimated probability is {0:5.3f}, exact is {1:5.3f}, estimated standard deviation is {2:5.3f}. Is this OK?".format(expected_p,p0,expected_error))
        
        estimated_log_marg_ll=logml_dict["logml"]
        
        # 3.2 corresponds to a bayes factor of 'Not worth more than a bare mention'
        self.assertTrue(np.abs(estimated_log_marg_ll-exact_log_marg_ll)<np.log(3.2),
                        msg="Estimated marginal log likelihood {0:2.5f}, exact marginal log likelihood {1:2.5f}. Is this OK?".format(estimated_log_marg_ll,exact_log_marg_ll)) 
        
    def test_gauss_gauss_gamma(self):
        """Testing the Bridge sampler with a Gauss-Gauss-Gamma model"""
        
        # prior parameters
        nu=np.random.random()*5.0
        S=np.random.random()*10.0
        beta=np.random.random()
        mu_prior=np.random.normal(0.0,10.0)
        
        draws=10000
        tune=1000
        n=10000
        
        print("Testing with nu=",nu,"S=",S,"mu_prior=",mu_prior)
        
        # random data
        mu0=np.random.normal(0.0,20.0)
        p=np.random.gamma(2.0,1.0)
        
        observations=np.random.normal(mu0,1.0/np.sqrt(p),size=n)
        
        with pymc3.Model() as GaussGaussGamma:
            
            prec=pymc3.Gamma('precision',alpha=nu,beta=1.0/S)
            mean=pymc3.Normal('mean',mu=mu_prior,tau=beta*prec)
            obs=pymc3.Normal('observations',mu=mean,tau=prec,observed=observations)
            trace=pymc3.sample(draws=draws,tune=tune)
        
        estimated_mean=trace["mean"].mean()
        estimated_precision=trace["precision"].mean()
        
        expected_error_mean=np.sqrt(1.0/(p*n))
        
        # should be true in 95% of cases
        self.assertTrue(np.abs(mu0-estimated_mean)<=2.0*expected_error_mean,
                        msg="Estimated mean {0:5.2f}, generating mean {1:5.2f}, expected error {2:6.3f}. Is this OK?".format(estimated_mean,mu0,expected_error_mean))
        
        # conjugate update for posterior parameters
        S_post= 1.0/(0.5* ((observations**2).sum()+beta*mu_prior**2 - (observations.sum()+beta*mu_prior)**2/(n+beta))+1.0/S)
        nu_post=nu+0.5*n
        
        expected_error_precision=np.sqrt(nu_post)*S_post
        
        estimated_precision2=nu_post*S_post
        
        # should be true in 95% of cases
        self.assertTrue(np.abs(p-estimated_precision)<=2.0*expected_error_precision,
                        msg="Estimated precision from samples {0:5.2f}, estimated from posterion {3:5.2f} generating precision {1:5.2f}, expected error {2:6.3f}. Is this OK?".format(estimated_precision,p,expected_error_precision,estimated_precision2))
        
        exact_log_marg_ll=0.5*np.log(beta/(beta+n))-0.5*n*np.log(2.0*np.pi) + (nu_post*np.log(S_post)+spf.gammaln(nu_post)) - (nu*np.log(S)+spf.gammaln(nu))
        
        # estimate with bridge sampling
        logml_dict=marginal_llk(trace, model=GaussGaussGamma, maxiter=10000)
        estimated_log_marg_ll=logml_dict["logml"]
        
         # 3.2 corresponds to a bayes factor of 'Not worth more than a bare mention'
        self.assertTrue(np.abs(estimated_log_marg_ll-exact_log_marg_ll)<np.log(3.2),
                        msg="Estimated marginal log likelihood {0:2.5f}, exact marginal log likelihood {1:2.5f}. Is this OK?".format(estimated_log_marg_ll,exact_log_marg_ll)) 


if __name__=="__main__":
    
    unittest.main()

