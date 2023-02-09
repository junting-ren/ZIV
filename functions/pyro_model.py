import gc
import pyro
import pyro.distributions as dist
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.zero_inflated import ZeroInflatedDistribution
from pyro.distributions.mixture import MaskedMixture
import torch.distributions.constraints as constraints
from pyro import poutine
from pyro.infer.autoguide import AutoLowRankMultivariateNormal,AutoDelta,AutoNormal,AutoContinuous,AutoLaplaceApproximation,AutoDiagonalNormal,init_to_median,init_to_feasible,init_to_value, AutoMultivariateNormal,init_to_mean,init_to_uniform,init_to_sample,AutoHierarchicalNormalMessenger, AutoGuideList
from pyro.poutine import block
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete,Trace_ELBO, Predictive, TraceMeanField_ELBO,JitTrace_ELBO
from pyro.distributions.torch_distribution import TorchDistributionMixin
from pyro.infer import MCMC, NUTS
from pyro.infer.mcmc.api import StreamingMCMC
from pyro.infer.mcmc.util import summary
import time
import random
from collections import defaultdict
import torch
import numpy as np
from pyro.infer import MCMC, NUTS,HMC
import matplotlib.pyplot as plt

from functools import partial

def spike_slab_model(relaxed, straight_through, temp , x, z):
    # Global variables.
    N, d = x.shape
    var_beta =  pyro.param('var_beta',  torch.tensor(1),constraints.positive)
    var_bias = pyro.param('var_beta_bias',  torch.tensor(0.5),constraints.positive)
    bias = pyro.sample('beta_bias', dist.Normal(loc=0, scale=torch.sqrt(var_bias)))
    #var_error = pyro.param('var_error',  torch.tensor(1), constraints.positive)
    var_error = pyro.sample('var_error',  dist.Gamma(0.1,0.1))
    pi = pyro.sample('pi', dist.Beta(1,5))
    if relaxed:
        if straight_through:
            delta = pyro.sample('delta', dist.RelaxedBernoulliStraightThrough(torch.tensor(temp), probs = pi).expand([d]).to_event(1))
        else:
            delta = pyro.sample('delta', dist.RelaxedBernoulli(torch.tensor(temp), probs = pi).expand([d]).to_event(1))
    else:
        delta = pyro.sample('delta', dist.Bernoulli(probs = pi).expand([d]).to_event(1))
    beta = pyro.sample('beta', dist.Normal(loc=0, scale=torch.sqrt(var_beta)).expand([d]).to_event(1))
    #import pdb;pdb.set_trace()
    latent_mean = (torch.mm(x, (beta*delta).unsqueeze(1).float())).squeeze(1)+bias
    var_genetic = torch.mean((latent_mean)**2) - torch.mean((latent_mean))**2
    #var_genetic = torch.sum(torch.abs(beta)**2)
    h = var_genetic/(var_genetic+var_error)
    # Samplingtorch.sum
    # with pyro.plate("data", x.shape[0]):
    #     pyro.sample("obs", tobit_dist(latent_mean, torch.sqrt(var_error)), obs = z)
    with pyro.plate("data", x.shape[0]):
        pyro.sample("obs", dist.Normal(latent_mean, torch.sqrt(var_error)), obs = z)
    #return zl_probit_mix_trun_norm_dist(p = model_prob, locs = locs, scale = scale_mix_trunc, start =start, end = end).sample()
    return h



def custom_guide(relaxed, straight_through, temp, x, z):
    d = x.shape[1]
    var_lambda_mu = pyro.param('var_lambda_mu', torch.tensor(0.), constraints.real)
    var_lambda_scale = pyro.param('var_lambda_scale', torch.tensor(0.5), constraints.positive)
    var_lambda =  pyro.sample('var_lambda', dist.LogNormal(var_lambda_mu, var_lambda_scale))
    
    bias_mu = pyro.param('bias_mu', torch.tensor(0.), constraints.real)
    bias_scale =  pyro.param('bais_scale', torch.tensor(0.5), constraints.positive)
    bias = pyro.sample('beta_bias', dist.Normal(loc=bias_mu, scale=bias_scale))
    
    pi_a = pyro.param('pi_a', torch.tensor(0.5), constraints.positive)
    pi_b = pyro.param('pi_b', torch.tensor(5.), constraints.positive)
    pi = pyro.sample('pi', dist.Beta(pi_a,pi_b))
    
    pi_local = pyro.param('local_pi', torch.rand(d), constraints.unit_interval)
    
    if relaxed:
        if straight_through:
            delta = pyro.sample('delta', dist.RelaxedBernoulliStraightThrough(torch.tensor(temp), 
                                                                              probs = pi_local).expand([d]).to_event(1))
        else:
            delta = pyro.sample('delta', dist.RelaxedBernoulli(torch.tensor(temp), 
                                                               probs = pi_local).expand([d]).to_event(1))
    else:
        delta = pyro.sample('delta', dist.Bernoulli(probs = pi_local).expand([d]).to_event(1))

    mu_local = pyro.param('mu_local', torch.randn(d)/torch.sqrt(torch.tensor(d)), constraints.real)
    scale_local = pyro.param('scale_local', torch.rand(d), constraints.positive)
    #scale_local2 = pyro.param('scale_local2', torch.ones(d)*0.01, constraints.positive)
    beta = pyro.sample('beta', dist.Normal(loc=mu_local, scale=scale_local).to_event(1))

    
def delta_guide(relaxed, straight_through, temp, x, z):
    d = x.shape[1]
    pi_local = pyro.param('local_pi', torch.rand(d), constraints.unit_interval)
    
    if relaxed:
        if straight_through:
            delta = pyro.sample('delta', dist.RelaxedBernoulliStraightThrough(torch.tensor(temp), 
                                                                              probs = pi_local).expand([d]).to_event(1))
        else:
            delta = pyro.sample('delta', dist.RelaxedBernoulli(torch.tensor(temp), 
                                                               probs = pi_local).expand([d]).to_event(1))
    else:
        delta = pyro.sample('delta', dist.Bernoulli(probs = pi_local).expand([d]).to_event(1))

        
def get_model(model, relaxed, straight_through, temp):
    return partial(model, relaxed, straight_through, temp)

def get_guide(model, guide_type, relaxed, straight_through, temp, p ):
    '''
    guide_type: character, 'low_rank_normal', 'diag_normal', 'custom', 'custom_delta'
    '''
    if guide_type == 'low_rank_normal':
        guide =  AutoLowRankMultivariateNormal(model,init_loc_fn=init_to_median, init_scale=1/p)
    elif guide_type == 'diag_normal':
        guide =  AutoDiagonalNormal(model,init_loc_fn=init_to_median, init_scale=1/p)
    elif guide_type == 'custom':
        guide = partial(custom_guide, relaxed, straight_through, temp)
    elif guide_type == 'custom_delta':
        guide = AutoGuideList(model)
        guide.append( AutoLowRankMultivariateNormal(block(model, hide = ['delta']),init_loc_fn=init_to_median, init_scale=1/p))
        guide.append(partial(delta_guide, relaxed, straight_through, temp))
    return guide

class experiment(object):
    def __init__(self, lr, z, X, model, guide, mcmc_index =  False, schedule_exp = 0.95, n_epoch = 30000, total_patience = 50, num_particles = 1, use_gpu = False, loss_mean_num = 200):
        self.model = model
        self.guide = guide
        self.loss_mean_num = loss_mean_num
        self.n_epoch = n_epoch
        self.total_patience = total_patience
        self.elbo = Trace_ELBO(num_particles =num_particles)
        #self.elbo = JitTrace_ELBO(num_particles =num_particles)
        #self.elbo = TraceMeanField_ELBO(num_particles =num_particles)
        self.mcmc_index = mcmc_index
        p = X.shape[1]
            
        # def per_param_args(param_name):
        #     if 'local_pi' in param_name:
        #         return {"lr": lr*5}
        #     else:
        #         return {"lr": lr}
        self.optim = pyro.optim.Adam({'lr': lr, 'betas': [0.95, 0.99]})
        #self.optim = pyro.optim.Adam(per_param_args)
        self.scheduler = pyro.optim.ExponentialLR({'optimizer': self.optim, 'optim_args': {'lr': lr}, 'gamma': schedule_exp})
        if use_gpu:
            cuda0 = torch.device('cuda:0')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.X, self.z = X.to(cuda0),z.to(cuda0)
            #num_cate = num_cate.to(cuda0)
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
            self.X, self.z = X.cpu(),z.cpu()
    def initialize(self, seed):
        pyro.set_rng_seed(seed)
        pyro.clear_param_store()
        self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo)
        return self.svi.loss(self.model, self.guide, self.X, self.z)
    def init_model(self):
        loss, seed = min((self.initialize(seed), seed) for seed in range(100))
        self.initialize(seed)
    def train(self, verbose = True, num_samples = 1000, warmup_steps=200):
        if self.mcmc_index:
            kernel = HMC(self.model, jit_compile  =True)
            #kernel = NUTS(self.model)
            self.mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps)
            self.mcmc.run(self.X, self.z)
        else:
            gradient_norms = defaultdict(list)
            for name, value in pyro.get_param_store().named_parameters():
                value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))
            losses = []
            inf_mean_loss = float('inf')
            patient = 0
            for i in range(self.n_epoch):
                loss = self.svi.step(self.X, self.z)
                losses.append(loss)
                if len(losses)>self.loss_mean_num:
                    mean_loss = np.mean(losses[-self.loss_mean_num:])
                    if inf_mean_loss <= mean_loss:
                        patient += 1
                    else:
                        inf_mean_loss = mean_loss
                        patient = 0
                self.scheduler.step()
                if verbose:
                    print(str(loss)+'\n' if (i % 100)==0 else '', end='')
                if patient > self.total_patience:
                    break
    def cal_heritability(self, num_samples = 500):
        if self.mcmc_index:
            #import pdb;pdb.set_trace()
            posterior = mcmc.get_samples().items()
        else:
            predictive = Predictive(self.model, guide=self.guide, num_samples=num_samples,
                            return_sites=("obs", "_RETURN", "beta", "delta", "var_error","pi"))
            samples = predictive(self.X, self.z)
            est = samples['_RETURN'].cpu().numpy().mean()
            std = samples['_RETURN'].cpu().numpy().std()
            upper = np.quantile(samples['_RETURN'].cpu().numpy(),0.975)
            lower = np.quantile(samples['_RETURN'].cpu().numpy(),0.025)
            self.samples = samples
            #import pdb; pdb.set_trace()
            #beta_star = dist.Bernoulli(probs= samples['pi'] ).sample()*samples['beta']
            # beta_star = samples['pi']*samples['beta']
            # emp_mean = torch.mm(beta_star, torch.t(self.X))
            # var_genetic = torch.var(emp_mean, dim = 1, keepdim = True)
            # var_error = samples['var_error']
            # h_est_v = var_genetic/(var_genetic+var_error)
            # est = np.mean(h_est_v.cpu().numpy())
            # std = h_est_v.cpu().numpy().std()
            # upper = np.quantile(h_est_v.cpu().numpy(),0.975)
            # lower = np.quantile(h_est_v.cpu().numpy(),0.025)
            #parameter_est = [self.guide.median()['var_beta_bias'],self.guide.median()['beta_bias']]
            # parameter_name = ['var_lambda', 'var_beta', 'var_beta_bias', 'beta_bias']
            #self.para_dict = self.guide.median()
            #import pdb;pdb.set_trace()
            # delta = self.para_dict["delta"].cpu().numpy()
            # self.beta_star = self.para_dict['beta']*delta
            #del para_dict["beta"]
            #del para_dict["lambda"]
            self.para_dict = None
            delta = samples['delta']
            beta = samples['beta']
            #import pdb; pdb.set_trace()
            self.beta_star = np.mean((delta*beta).detach().cpu().numpy(), axis = 0).squeeze(0)
            return est, std, lower, upper,self.para_dict
    def plot_betas(self, true_beta):
        M = len(true_beta)
        fig = plt.figure(figsize=(16,8), facecolor='white')
        ax = fig.add_subplot(1,1,1)
        ax.plot(np.arange(M), true_beta, \
               linewidth = 3, color = "black", label = "ground truth")
        ax.scatter(np.arange(M), true_beta, \
               s = 70, marker = '+', color = "black")
        ax.plot(np.arange(M), self.beta_star, \
                   linewidth = 3, color = "red", \
                   label = "linear model with spike and slab prior")
        ax.set_xlim([0,M-1])
        ax.set_ylabel("Slopes", fontsize=18)
        ax.hlines(0,0,M-1)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.legend(prop={'size':14})

        fig.set_tight_layout(True)
        fig.savefig('foo.png')
        plt.show()
    def clear_cache(self):
        gc.collect()
        torch.cuda.empty_cache()