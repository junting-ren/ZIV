import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


class linear_slab_spike(nn.Module):
    def __init__(self, p, init_pi_local = 0.45, init_pi_global = 0.5, init_beta_var = 1, init_noise_var = 1,
                gumbel_softmax_temp = 0.5, gumbel_softmax_hard = False, a= 1.1,b=3.1, init_c= 1.1, init_d = 5.1,
                q1 = 1.1, q2 = 1.1, init_q3 = 1.1, init_q4 = 1.1, n_E = 50):
        super(linear_slab_spike, self).__init__()
        self.p = p
        prior_uni = np.sqrt(1/p)
        #prior_uni = 1/p
        # Variational parameters
        self.beta_mu = nn.Parameter(torch.FloatTensor(size = (p,)).uniform_(-prior_uni,prior_uni)) # beta mean
        self.beta_log_var = nn.Parameter(torch.log(torch.rand((p,)))) # beta log variance
        self.logit_pi_local = nn.Parameter(torch.logit(torch.FloatTensor(size = (p,)).uniform_(init_pi_local-0.05,init_pi_local+0.05))) # beta local pi on the logit scale
        self.log_c = nn.Parameter(torch.log(torch.tensor((init_c,))))
        self.log_d =  nn.Parameter(torch.log(torch.tensor((init_d,))))
        self.log_q3 = nn.Parameter(torch.log(torch.tensor((init_q3,))))
        self.log_q4 = nn.Parameter(torch.log(torch.tensor((init_q4,))))
        # MLE parameters
        self.bias = nn.Parameter(torch.tensor((1.,)))
        self.logit_pi_global = nn.Parameter(torch.logit(torch.tensor(init_pi_global))) # global pi on logit scale 
        self.beta_log_var_prior = nn.Parameter(torch.log(torch.tensor(init_beta_var))) # beta prior log variance
        self.log_var_noise = nn.Parameter(torch.log(torch.tensor(init_noise_var))) # linear noise prior variance
        # prior parameters for global pi
        self.a = torch.tensor((a,))
        self.b = torch.tensor((b,))
        #Priors parameters for noise variance
        self.q1 = torch.tensor((q1,))
        self.q2 = torch.tensor((q2,))
        # gumbel hyper parameters
        self.tau = gumbel_softmax_temp
        self.hard = gumbel_softmax_hard
        #number of samples for empirical integration
        self.n_E = n_E
    def get_para_orig_scale(self):
        return torch.exp(self.log_var_noise),torch.exp(self.beta_log_var), torch.sigmoid(self.logit_pi_local), \
                torch.exp(self.beta_log_var_prior), \
                torch.exp(self.log_c), torch.exp(self.log_d),\
                torch.exp(self.log_q3),torch.exp(self.log_q4)
    
    def log_data_lh(self, beta, delta, X, y, q3, q4):
        n = X.shape[0]
        est_mean = (beta*delta) @ X.t()+self.bias
        #import pdb;pdb.set_trace()
        return torch.mean(-n*0.5*(torch.log(q4)-torch.digamma(q3))-1/(2)*q3/q4*torch.sum(torch.square(y-est_mean),dim = 1))
    
    def log_prior_expect_lh(self, pi_local, beta_var, beta_var_prior, c, d, q3, q4):
        # expectation of log pi
        expect_lpi = torch.digamma(c)-torch.digamma(c + d)
        # expectation of log(1-pi)
        expect_l1_pi = torch.digamma(d)-torch.digamma(c + d)
        # first part 
        lh_1 = torch.sum(expect_lpi*pi_local\
        + expect_l1_pi*(1. - pi_local)) \
        - self.p*0.5*self.beta_log_var_prior \
        - 0.5*(torch.sum(beta_var)+torch.sum(torch.square(self.beta_mu)))/beta_var_prior
        # expected log likelihood of global pi part
        lh_global_pi = (self.a-1)*expect_lpi+(self.b - 1)* expect_l1_pi - torch.lgamma(self.a)-torch.lgamma(self.b)+torch.lgamma(self.a+self.b)
        # expected log likelihood of noise variance
        lh_noise_var = self.q1*torch.log(self.q2)-torch.lgamma(self.q1)\
        -(self.q1+1)*(torch.log(q4)-torch.digamma(q3)) - self.q2*q3/q4
        return lh_1+lh_global_pi+lh_noise_var
    
    def log_entropy(self, pi_local, c, d, q3, q4):
        entropy1 = -torch.sum(
            pi_local*torch.log(pi_local)-0.5*self.beta_log_var + (1-pi_local)*torch.log(1-pi_local)
        )
        entropy_global_pi = torch.lgamma(c)+torch.lgamma(d) - torch.lgamma(c+d) - (c - 1)*torch.digamma(c)\
                            - (d - 1)*torch.digamma(d)+(c+d-2)*torch.digamma(c+d)
        entropy_noise_var = q3+torch.log(q4)+torch.lgamma(q3)-(q3+1)*torch.digamma(q3)
        return entropy1+entropy_global_pi+entropy_noise_var
    
    def ELBO(self,X, y):
        # get the current parameter after transformation
        noise_var, beta_var, pi_local,beta_var_prior,c, d, q3, q4 = self.get_para_orig_scale()
        # reparameterization
        #import pdb; pdb.set_trace()
        beta = self.beta_mu + torch.sqrt(beta_var)*torch.randn((self.n_E,self.p))
        # Gumbel-softmax sampling
        delta = nn.functional.gumbel_softmax(torch.stack( [ self.logit_pi_local.expand(self.n_E ,-1), -self.logit_pi_local.expand(self.n_E,-1) ], dim = 2 ),dim = 2, tau = self.tau, hard = self.hard)[:,:,0]
        # ELBO
        ELBO = self.log_data_lh(beta, delta, X, y, q3, q4) + \
            self.log_prior_expect_lh(pi_local, beta_var, beta_var_prior, c, d, q3, q4 ) + \
            self.log_entropy(pi_local, c, d, q3, q4 )
        return ELBO
    
    def inference(self, X, num_samples = 500, plot = False, true_beta = None):
        beta_mean = (self.beta_mu.detach()).numpy()
        beta_std = torch.exp(self.beta_log_var).detach().numpy()
        pi_local = torch.sigmoid(self.logit_pi_local.detach()).numpy()
        # global pi posterior parameters
        c = np.exp(self.log_c.detach().numpy())
        d = np.exp(self.log_d.detach().numpy())
        # global pi posterior
        global_pi_poster = np.random.beta(c,d, size = (num_samples,))
        #import pdb; pdb.set_trace()
        delta = np.random.binomial(n = 1, p = pi_local, size = (num_samples, self.p))
        sample_beta = np.random.normal(loc = beta_mean, scale = beta_std, size = (num_samples, self.p))*delta # num_samples* p
        est_mean = X.numpy() @ np.transpose(sample_beta) + self.bias.detach().numpy() # a n*num_samples matrix
        # Noise variance poseterior parameters
        q3 = np.exp(self.log_q3.detach().numpy())
        q4 = np.exp(self.log_q4.detach().numpy())
        # Noise variance posterior
        noise_var_poster = 1/np.random.gamma(q3,1/q4, size = (num_samples,))
        noise_var_est = np.mean(noise_var_poster)
        # posterior for h
        var_genetic_est = np.mean(est_mean**2, axis = 0) - np.mean(est_mean, axis = 0)**2
        var_genetic_mean = np.mean(var_genetic_est)
        #import pdb; pdb.set_trace()
        h_est = var_genetic_est/(var_genetic_est+noise_var_poster) # s*1 vector
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
                'mean_var_genetic': var_genetic_mean, 'noise_var': noise_var_est, 
                'global_pi':global_pi_est, 'global_pi_upper':global_pi_upper, 'global_pi_lower':global_pi_lower
               }
        