from jax import random
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS,SA,HMCECS, init_to_median, MixedHMC, HMC,DiscreteHMCGibbs,HMCGibbs
import numpyro
import jax.numpy as jnp
import numpy as np
from tobit_dist import tobit_dist
class experiment(object):
    def __init__(self, model, z, X,use_gpu = False, tobit = False):
        self.model = model
        self.z = z
        self.X = X
        self.p = X.shape[1]
    def train(self, step_size, verbose = True, num_samples = 1000, warmup_steps=200):
        kernel = DiscreteHMCGibbs(NUTS(self.model,step_size = step_size, init_strategy  = init_to_median))
        #kernel = HMCGibbs(inner_kernel = NUTS(self.model,step_size = step_size, init_strategy  = init_to_median), gibbs_fn=gibbs_fn, gibbs_sites=["beta"])
        self.mcmc = MCMC(kernel, num_warmup=warmup_steps, num_samples=num_samples)
        self.mcmc.run(random.PRNGKey(0),jnp.asarray(self.X),jnp.asarray(self.z))
    def cal_heritability(self):
        #import pdb; pdb.set_trace()
        posterior = self.mcmc.get_samples()
        beta = posterior["beta"]* posterior["delta"]
        g_var = (beta @ jnp.transpose(self.X) ).var(axis = 1)
        dat_var = posterior["var_error"]
        heritability = g_var/(dat_var+g_var)
        return heritability,g_var,dat_var
    
def linear_mcmc_model(x, z):
    # Global variables.
    N, d = x.shape
    #var_beta =  pyro.sample('var_beta', dist.InverseGamma(5, 0.1))
    var_beta =  numpyro.sample('var_beta', dist.InverseGamma(1, 1))
    #var_lambda =  numpyro.sample('var_lambda', dist.InverseGamma(1, 1))
    pi =  numpyro.sample('pi', dist.Beta(0.5, 0.5))
    var_bias = numpyro.sample('var_beta_bias', dist.InverseGamma(1,1))
    bias = numpyro.sample('beta_bias', dist.Normal(loc=0, scale=jnp.sqrt(var_bias)))
    var_error = numpyro.sample('var_error', dist.InverseGamma(0.1,0.1))
    #c = torch.tensor(1)
    with numpyro.plate('local_shrinkage', d):
        #Lambda = numpyro.sample('lambda', dist.HalfCauchy(scale=jnp.sqrt(var_lambda)))
        #Lambda = pyro.sample('lambda', dist.HalfCauchy(scale=1))
        # Traditional horseshoe
        #horseshoe_sigma = jnp.sqrt(var_beta)*Lambda
        horseshoe_sigma = jnp.sqrt(var_beta)
        #Regularized horseshoe
        # scale_beta = torch.sqrt(var_beta)
        # horseshoe_sigma = torch.sqrt(c**2*Lambda**2/(c**2+scale_beta**2*Lambda**2))*scale_beta
        delta = numpyro.sample('delta', dist.Bernoulli(probs=pi))
        beta = numpyro.sample('beta', dist.Normal(loc=0, scale=horseshoe_sigma))
    #import pdb; pdb.set_trace()
    latent_mean = jnp.dot(x, beta*delta)+bias
    #y_star = numpyro.sample('y_star',dist.TruncatedNormal(loc = latent_mean, scale = jnp.sqrt(var_error), high = jnp.where(z==0, 0, float('Inf')) ) )
    var_genetic = jnp.mean((latent_mean)**2) - jnp.mean((latent_mean))**2
    #var_genetic = torch.sum(torch.abs(beta)**2)
    h = var_genetic/(var_genetic+var_error)
    # Sampling
    # with numpyro.plate("data", x.shape[0]):
    #     numpyro.sample("obs", tobit_dist(latent_mean, jnp.sqrt(var_error)), obs = z)
    with numpyro.plate("data", x.shape[0]):
        numpyro.sample("obs", dist.Normal(latent_mean, jnp.sqrt(var_error)), obs = z)
    return h


def tobit_mcmc_model(x, z):
    # Global variables.
    N, d = x.shape
    #var_beta =  pyro.sample('var_beta', dist.InverseGamma(5, 0.1))
    var_beta =  numpyro.sample('var_beta', dist.InverseGamma(1, 1))
    #var_lambda =  numpyro.sample('var_lambda', dist.InverseGamma(1, 1))
    pi =  numpyro.sample('pi', dist.Beta(0.5, 0.5))
    var_bias = numpyro.sample('var_beta_bias', dist.InverseGamma(1,1))
    bias = numpyro.sample('beta_bias', dist.Normal(loc=0, scale=jnp.sqrt(var_bias)))
    var_error = numpyro.sample('var_error', dist.InverseGamma(0.1,0.1))
    #c = torch.tensor(1)
    with numpyro.plate('local_shrinkage', d):
        #Lambda = numpyro.sample('lambda', dist.HalfCauchy(scale=jnp.sqrt(var_lambda)))
        #Lambda = pyro.sample('lambda', dist.HalfCauchy(scale=1))
        # Traditional horseshoe
        #horseshoe_sigma = jnp.sqrt(var_beta)*Lambda
        horseshoe_sigma = jnp.sqrt(var_beta)
        #Regularized horseshoe
        # scale_beta = torch.sqrt(var_beta)
        # horseshoe_sigma = torch.sqrt(c**2*Lambda**2/(c**2+scale_beta**2*Lambda**2))*scale_beta
        delta = numpyro.sample('delta', dist.Bernoulli(probs=pi))
        beta = numpyro.sample('beta', dist.Normal(loc=0, scale=horseshoe_sigma))
    #import pdb; pdb.set_trace()
    latent_mean = jnp.dot(x, beta*delta)+bias
    #y_star = numpyro.sample('y_star',dist.TruncatedNormal(loc = latent_mean, scale = jnp.sqrt(var_error), high = jnp.where(z==0, 0, float('Inf')) ) )
    var_genetic = jnp.mean((latent_mean)**2) - jnp.mean((latent_mean))**2
    #var_genetic = torch.sum(torch.abs(beta)**2)
    h = var_genetic/(var_genetic+var_error)
    # Sampling
    # with numpyro.plate("data", x.shape[0]):
    #     numpyro.sample("obs", tobit_dist(latent_mean, jnp.sqrt(var_error)), obs = z)
    with numpyro.plate("data", x.shape[0]):
        numpyro.sample("obs", tobit_dist(latent_mean, jnp.sqrt(var_error)), obs = z)
    return h