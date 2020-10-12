
import unittest

import numpy as np
import pymc3 as pm
import scipy.special as spf

from neuropsymodelcomparison.dataprocessing.bridgesampler import marginal_llk

class TestBridgeSampler(unittest.TestCase):
    
    def test_bernoulli_process(self):
        """Testing the Bridge Sampler with a Beta-Bernoulli-Process model"""
        
        # prior parameters
        alpha = np.random.gamma(1.0, 2.0)
        beta = np.random.gamma(1.0, 2.0)
        n = 100
        
        draws = 10000
        tune = 1000
        
        print("Testing with alpha = ", alpha, "and beta = ", beta)
        
        # random data
        p0 = np.random.random()
        expected_error = np.sqrt(p0*(1-p0)/n) # reasonable approximation
        
        observations = (np.random.random(n)<=p0).astype("int")
        
        with pm.Model() as BernoulliBeta:
    
            theta = pm.Beta('pspike', alpha=alpha, beta=beta)
            obs = pm.Categorical('obs', p=pm.math.stack([theta, 1.0-theta]), observed=observations)
            trace = pm.sample(draws=draws, tune=tune)

        # calculate exact marginal likelihood
        n = len(observations)
        k = sum(observations)
        print(n, k)
        exact_log_marg_ll = spf.betaln(alpha+k, beta+(n-k)) - spf.betaln(alpha, beta)
    
        # estimate with bridge sampling
        logml_dict = marginal_llk(trace, model=BernoulliBeta, maxiter=10000)
        expected_p = 1.0-trace["pspike"].mean()
        
        # should be true in 95% of the runs
        self.assertTrue(np.abs(expected_p-p0)<2*expected_error, 
                        msg="Estimated probability is {0:5.3f}, exact is {1:5.3f}, estimated standard deviation is {2:5.3f}. Is this OK?".format(expected_p, p0, expected_error))
        
        estimated_log_marg_ll = logml_dict["logml"]
        
        # 3.2 corresponds to a bayes factor of 'Not worth more than a bare mention'
        self.assertTrue(np.abs(estimated_log_marg_ll-exact_log_marg_ll)<np.log(3.2), 
                        msg="Estimated marginal log likelihood {0:2.5f}, exact marginal log likelihood {1:2.5f}. Is this OK?".format(estimated_log_marg_ll, exact_log_marg_ll)) 
        
    def test_gauss_gauss_gamma(self):
        """Testing the Bridge sampler with a Gauss-Gauss-Gamma model"""
        
        # prior parameters
        nu = np.random.random()*5.0
        S = np.random.random()*10.0
        beta = np.random.random()
        mu_prior = np.random.normal(0.0, 10.0)
        
        draws = 10000
        tune = 1000
        n = 10000
        
        print("Testing with nu = ", nu, "S = ", S, "mu_prior = ", mu_prior)
        
        # random data
        mu0 = np.random.normal(0.0, 20.0)
        p = np.random.gamma(2.0, 1.0)
        
        observations = np.random.normal(mu0, 1.0/np.sqrt(p), size=n)
        
        with pm.Model() as GaussGaussGamma:
            
            prec = pm.Gamma('precision', alpha=nu, beta=1.0/S)
            mean = pm.Normal('mean', mu=mu_prior, tau=beta*prec)
            obs = pm.Normal('observations', mu=mean, tau=prec, observed=observations)
            trace = pm.sample(draws=draws, tune=tune)
        
        estimated_mean = trace["mean"].mean()
        estimated_precision = trace["precision"].mean()
        
        expected_error_mean = np.sqrt(1.0/(p*n))
        
        # should be true in 95% of cases
        self.assertTrue(np.abs(mu0-estimated_mean)<=2.0*expected_error_mean, 
                        msg="Estimated mean {0:5.2f}, generating mean {1:5.2f}, expected error {2:6.3f}. Is this OK?".format(estimated_mean, mu0, expected_error_mean))
        
        # conjugate update for posterior parameters
        S_post = 1.0/(0.5* ((observations**2).sum()+beta*mu_prior**2 - (observations.sum()+beta*mu_prior)**2/(n+beta))+1.0/S)
        nu_post = nu+0.5*n
        
        expected_error_precision = np.sqrt(nu_post)*S_post
        
        estimated_precision2 = nu_post*S_post
        
        # should be true in 95% of cases
        self.assertTrue(np.abs(p-estimated_precision)<=2.0*expected_error_precision, 
                        msg="Estimated precision from samples {0:5.2f}, estimated from posterion {3:5.2f} generating precision {1:5.2f}, expected error {2:6.3f}. Is this OK?".format(estimated_precision, p, expected_error_precision, estimated_precision2))
        
        exact_log_marg_ll = 0.5*np.log(beta/(beta+n))-0.5*n*np.log(2.0*np.pi) + (nu_post*np.log(S_post)+spf.gammaln(nu_post)) - (nu*np.log(S)+spf.gammaln(nu))
        
        # estimate with bridge sampling
        logml_dict = marginal_llk(trace, model=GaussGaussGamma, maxiter=10000)
        estimated_log_marg_ll = logml_dict["logml"]
        
         # 3.2 corresponds to a bayes factor of 'Not worth more than a bare mention'
        self.assertTrue(np.abs(estimated_log_marg_ll-exact_log_marg_ll)<np.log(3.2), 
                        msg="Estimated marginal log likelihood {0:2.5f}, exact marginal log likelihood {1:2.5f}. Is this OK?".format(estimated_log_marg_ll, exact_log_marg_ll))


if __name__=="__main__":
    
    unittest.main()
