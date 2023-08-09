# This class works
import numpy as np
import pandas as pd
import numpyro.distributions as dist
import numpyro
import jax.numpy as jnp

class tobit_dist(numpyro.distributions.Distribution):
    arg_constraints = {}
    def __init__(self, mean, sigma):
        #import pdb; pdb.set_trace()
        self.loc = mean
        self.scale = sigma
        batch_shape = len(self.loc)
        super().__init__((batch_shape,))
    def sample(self,  key, sample_shape = ()):
        #import pdb; pdb.set_trace()
        # y_star = dist.Normal(loc = self.loc, scale = self.scale).sample(sample_shape = sample_shape)
        # sample = jnp.where(y_star<=0, jnp.zeros(y_star.shape).double(),y_star)
        # return sample
        pass
    def log_prob(self, value):
        #import pdb; pdb.set_trace()
        l_p1 = jnp.log(dist.Normal(loc = 0,  scale = 1).cdf(-self.loc/self.scale))*(value==0)
        l_p2 = dist.Normal(loc = 0,  scale = self.scale).log_prob((value-self.loc))*(value>0)
        l_p = l_p1+l_p2
        # l_p =jnp.where(value==0, 
        #                  jnp.log(dist.Normal(loc = 0,  scale = 1).cdf(-self.loc/self.scale)+0.00001), 
        #                  dist.Normal(loc = 0,  scale = self.scale).log_prob((value-self.loc)) 
        #                                )
        return l_p