import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


class linear_slab_spike(nn.Module):
    def __init__(self, p, init_pi_local = 0.45, init_pi_global = 0.5, init_beta_var = 1, init_noise_var = 1,
                gumbel_softmax_temp = 0.5, gumbel_softmax_hard = False, a= 1.1,b=3.1, init_c= 1.1, init_d = 5.1):
        super(linear_slab_spike, self).__init__()
        self.p = p
        prior_uni = np.sqrt(1/p)
        #prior_uni = 1/p
        # Variational parameters
        self.beta_mu = nn.Parameter(torch.FloatTensor(size = (p,)).uniform_(-prior_uni,prior_uni)) # beta mean
        self.beta_log_var = nn.Parameter(torch.log(torch.rand((p,)))) # beta log variance
        self.logit_pi_local = nn.Parameter(torch.logit(torch.FloatTensor(size = (p,)).uniform_(init_pi_local-0.05,init_pi_local+0.05))) # beta local pi on the logit scale
        self.c = nn.Parameter(torch.tensor((init_c,)))
        self.d =  nn.Parameter(torch.tensor((init_d,)))
        # MLE parameters
        self.bias = nn.Parameter(torch.tensor((1.,)))
        self.logit_pi_global = nn.Parameter(torch.logit(torch.tensor(init_pi_global))) # global pi on logit scale 
        self.beta_log_var_prior = nn.Parameter(torch.log(torch.tensor(init_beta_var))) # beta prior log variance
        self.log_var_noise = nn.Parameter(torch.log(torch.tensor(init_noise_var))) # linear noise prior variance
        # prior parameters for global pi
        # self.a = nn.Parameter(torch.tensor((1.1,)))
        # self.b = nn.Parameter(torch.tensor((3.,)))
        self.a = torch.tensor((a,))
        self.b = torch.tensor((b,))
        
        # gumbel hyper parameters
        self.tau = gumbel_softmax_temp
        self.hard = gumbel_softmax_hard
        
    def get_para_orig_scale(self):
        return torch.exp(self.beta_log_var), torch.sigmoid(self.logit_pi_local), \
                torch.sigmoid(self.logit_pi_global), torch.exp(self.beta_log_var_prior), torch.exp(self.log_var_noise)
    
    def log_data_lh(self, beta, delta, X, y, var_noise):
        n = X.shape[0]
        est_mean = (beta*delta) @ X.t()+self.bias
        return -n*0.5*self.log_var_noise-1/(2*var_noise)*torch.sum(torch.square(y-est_mean))
    
    def log_prior_expect_lh(self, pi_global, pi_local, beta_var, beta_var_prior):
        # expectation of log pi
        expect_lpi = torch.digamma(self.c)-torch.digamma(self.c + self.d)
        # expectation of log(1-pi)
        expect_l1_pi = torch.digamma(self.d)-torch.digamma(self.c + self.d)
        # first part 
        lh_1 = torch.sum(expect_lpi*pi_local\
        + expect_l1_pi*(1. - pi_local)) \
        - self.p*0.5*self.beta_log_var_prior \
        - 0.5*(torch.sum(beta_var)+torch.sum(torch.square(self.beta_mu)))/beta_var_prior
        # expected log likelihood of global pi part
        lh_global_pi = (self.a-1)*expect_lpi+(self.b - 1)* expect_l1_pi - torch.lgamma(self.a)-torch.lgamma(self.b)+torch.lgamma(self.a+self.b)
        return lh_1+lh_global_pi
    
    def log_entropy(self, pi_local):
        entropy1 = -torch.sum(
            pi_local*torch.log(pi_local)-0.5*self.beta_log_var + (1-pi_local)*torch.log(1-pi_local)
        )
        entropy_global_pi = torch.lgamma(self.c)+torch.lgamma(self.d) - torch.lgamma(self.c+self.d) - (self.c - 1)*torch.digamma(self.c)\
                            - (self.d - 1)*torch.digamma(self.d)+(self.c+self.d-2)*torch.digamma(self.c+self.d)
        return entropy1+entropy_global_pi
    
    def ELBO(self,X, y):
        # get the current parameter after transformation
        beta_var, pi_local, pi_global, beta_var_prior, var_noise = self.get_para_orig_scale()
        # reparameterization
        beta = self.beta_mu + torch.sqrt(beta_var)*torch.randn((self.p,))
        # Gumbel-softmax sampling
        delta = nn.functional.gumbel_softmax(torch.column_stack( [ self.logit_pi_local, -self.logit_pi_local ] ),dim = 1, tau = self.tau, hard = self.hard)[:,0]
        # ELBO
        ELBO = self.log_data_lh(beta, delta, X, y, var_noise) + \
            self.log_prior_expect_lh(pi_global, pi_local, beta_var, beta_var_prior) + self.log_entropy(pi_local)
        return ELBO
    
    def inference(self, X, num_samples = 500, plot = False, true_beta = None):
        beta_mean = (self.beta_mu.detach()).numpy()
        beta_std = torch.exp(self.beta_log_var).detach().numpy()
        pi_local = torch.sigmoid(self.logit_pi_local.detach()).numpy()
        # global pi posterior parameters
        c = self.c.detach().numpy()
        d = self.d.detach().numpy()
        # global pi posterior
        global_pi_poster = np.random.beta(c,d, size = (num_samples,))
        #import pdb; pdb.set_trace()
        delta = np.random.binomial(n = 1, p = pi_local, size = (num_samples, self.p))
        sample_beta = np.random.normal(loc = beta_mean, scale = beta_std, size = (num_samples, self.p))*delta # num_samples* p
        est_mean = X.numpy() @ np.transpose(sample_beta) + self.bias.detach().numpy() # a n*num_samples matrix
        noise_var = torch.exp(self.log_var_noise.detach()).numpy()
        var_genetic_est = np.mean(est_mean**2, axis = 0) - np.mean(est_mean, axis = 0)**2
        var_genetic_mean = np.mean(var_genetic_est)
        h_est = var_genetic_est/(var_genetic_est+noise_var) # s*1 vector
        mean_h_est = np.mean(h_est)
        upper = np.quantile(h_est, q = 0.975)
        lower = np.quantile(h_est, q = 0.025)
        global_pi_est = np.mean(global_pi_poster)
        global_pi_upper = np.quantile(global_pi_poster, q = 0.975)
        global_pi_lower = np.quantile(global_pi_poster, q = 0.025)
        if plot and true_beta is not None:
            fig = plt.figure(figsize=(16,8), facecolor='white')
            ax = fig.add_subplot(1,1,1)
            ax.plot(np.arange(self.p), true_beta, \
                   linewidth = 3, color = "black", label = "ground truth")
            ax.scatter(np.arange(self.p), true_beta, \
                   s = 70, marker = '+', color = "black")
            ax.plot(np.arange(self.p),  beta_mean, \
                       linewidth = 3, color = "red", \
                       label = "linear model with spike and slab prior")
            ax.set_xlim([0,self.p-1])
            ax.set_ylabel("Slopes", fontsize=18)
            ax.hlines(0,0,self.p-1)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().set_visible(False)
            ax.legend(prop={'size':14})
            fig.set_tight_layout(True)
            plt.show()
        return {'mean_h_est': mean_h_est, 'h_est_upper': upper, 'h_est_lower': lower, 
                'mean_var_genetic': var_genetic_mean, 'noise_var': noise_var, 
                'global_pi':global_pi_est, 'global_pi_upper':global_pi_upper, 'global_pi_lower':global_pi_lower
               }
        