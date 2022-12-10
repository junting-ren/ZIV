import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# def cust_act(x):
#     return (x/torch.sqrt(torch.sqrt(1+x**4))+1)*0.5
def cust_act(x):
    return torch.sigmoid(x)
# def cust_act(x):
#     return (x/torch.sqrt(1+x**2)+1)*0.5
# def cust_act(x):
#     return (x/(1+torch.abs(x))+1)*0.5
# def cust_act(x):
#     return (2/3.14*torch.arctan(3.14/2*x)+1)*0.5
# def cust_act(x):
#     return (torch.tanh(x)+1)*0.5
class linear_slab_spike(nn.Module):
    def __init__(self, p, init_pi_local = 0.45, init_pi_global = 0.5, init_beta_var = 1, init_noise_var = 1,
                gumbel_softmax_temp = 0.5, gumbel_softmax_hard = False, a1= 1.1,a2=3.1, init_a3= 1.1, init_a4 = 5.1,
                q1 = 1.1, q2 = 1.1, init_q3 = 1.1, init_q4 = 1.1, n_E = 50, prior_sparsity = False):
        '''Initialize fast variational inference Bayesian slab and spike linear model
        
        Parameters:
        ------------------------
        p: number of features, not including the bias
        init_pi_local: the initial value of sparsity probability for each feature
        init_pi_global:  the initial value of global sparsity probability (MLE), not used when prior_sparsity is True
        init_beta_var: the initial value of feature coefficent variance
        init_noise_var: inital value for noise variance
        gumbel_softmax_temp: gumbel softmax temperature value, the smaller the value, the more closer to the true binary sample but with greater graident variance
        gumbel_softmax_hard:  if True, the returned samples will be discretized as one-hot vectors, but will be differentiated as if it is the soft sample in autograd
        a1: prior beta distribution parameters for global sparsity
        a2: prior beta distribution parameters for global sparsity
        init_a3: posterior beta distribution parameter
        init_a4:posterior beta distribution parameter
        q1: prior inverse gamma distribution parameters for noise varaiance
        q2: prior inverse gamma distribution parameters for noise varaiance
        init_q3: posterior inverse gamma distribution parameters for noise varaiance
        init_q4: posterior inverse gamma distribution parameters for noise varaiance
        n_E: number of samples for the empirical expectation of the data likelihood
        prior_sparsity: If True, the global sparsity will have a prior or else it will be estimated through Maximum likelihood.
        
        '''
        super(linear_slab_spike, self).__init__()
        # Fixed values in the model
        self.p = p #number of features
        prior_uni = np.sqrt(1/p) # initial values for coefficient mean 
        self.a1 = torch.tensor((a1,))# prior parameters for global pi
        self.a2 = torch.tensor((a2,))# prior parameters for global pi
        self.q1 = torch.tensor((q1,))#Priors parameters for noise variance
        self.q2 = torch.tensor((q2,))#Priors parameters for noise variance
        self.tau = gumbel_softmax_temp# gumbel hyper parameters        
        self.hard = gumbel_softmax_hard# gumbel hyper parameters
        self.n_E = n_E #number of samples for empirical integration
        self.prior_sparsity = prior_sparsity # If True, the global sparsity will have a prior or else it will be estimated through Maximum likelihood.
        # Variational approximation posterior distribution parameters
        self.beta_mu = nn.Parameter(torch.FloatTensor(size = (p,)).uniform_(-prior_uni,prior_uni)) # beta mean:Xavier Uniform Distribution initilization
        self.beta_log_var = nn.Parameter(torch.log(torch.rand((p,)))) # beta log variance: uniform 0,1 initilizaiton
        self.logit_pi_local = nn.Parameter(torch.logit(torch.FloatTensor(size = (p,)).uniform_(init_pi_local-0.05,init_pi_local+0.05))) # sparsity for each feature, uniform initilized
        self.log_a3 = nn.Parameter(torch.log(torch.tensor((init_a3,)))) # prior for global pi beta distribution
        self.log_a4 =  nn.Parameter(torch.log(torch.tensor((init_a4,))))# prior for global pi beta distribution
        self.log_q3 = nn.Parameter(torch.log(torch.tensor((init_q3,))))# prior for noise variance inverse gamma distribution
        self.log_q4 = nn.Parameter(torch.log(torch.tensor((init_q4,))))# prior for noise variance inverse gamma distribution
        # MLE parameters
        self.bias = nn.Parameter(torch.tensor((1.,)))# Bias
        self.logit_pi_global = nn.Parameter(torch.logit(torch.tensor(init_pi_global))) # global pi on logit scale 
        self.beta_log_var_prior = nn.Parameter(torch.log(torch.tensor(init_beta_var))) # beta prior log variance
        self.log_var_noise = nn.Parameter(torch.log(torch.tensor(init_noise_var))) # linear noise prior variance
        
    def get_para_orig_scale(self):
        '''Function to calculate the parameters ont he original scale
        '''
        return torch.exp(self.log_var_noise),torch.exp(self.beta_log_var), cust_act(self.logit_pi_local), \
                torch.exp(self.beta_log_var_prior), \
                torch.exp(self.log_a3), torch.exp(self.log_a4),\
                torch.exp(self.log_q3),torch.exp(self.log_q4), cust_act(self.logit_pi_global)
    
    def log_data_lh(self, beta, delta, X, y, q3, q4):
        '''Calculate the expected data likelihood over the approximation posterior
        
        Parameters:
        ----------------------
        beta: sampled coefficient, n_E by p matrix
        delta: sampled sparsity, n_E by p matrix
        X: feature matrix, n by p
        y: outcome vector, n by 1
        q3: noise variance inverse gamma for approximation posterior
        q4: noise variance inverse gamma for approximation posterior
        
        Return:
        ----------------------
        Expected data likelihood over the approximation posterior
        '''
        n = X.shape[0]
        est_mean = (beta*delta) @ X.t()+self.bias
        #import pdb;pdb.set_trace()
        return torch.mean(-n*0.5*(torch.log(q4)-torch.digamma(q3))-1/(2)*q3/q4*torch.sum(torch.square(y-est_mean),dim = 1))
    
    def log_prior_expect_lh(self, pi_local, beta_var, beta_var_prior, a3, a4, q3, q4, pi_global):
        '''Calculate the expected data likelihood over the approximation posterior
        
        Parameters:
        ------------------
        pi_local: sparsity parameter for approximation posterior: p by 1
        beta_var: variance for coefficient for approximation posterior: p by 1
        beta_var_prior: fixed prior variance for coefficient
        a3: global sparsity beta distribution parameter for approximation posterior
        a4: global sparsity beta distribution parameter for approximation posterior
        q3: noise variance inverse gamma for approximation posterior
        q4: noise variance inverse gamma for approximation posterior
        pi_global: global sparsity if prior_sparsity=False, or else it is not used
        
        Return:
        -------------------
        Expected prior likelihood over the approximation posterior
        '''
        # expected log likelihood of noise variance
        lh_noise_var = self.q1*torch.log(self.q2)-torch.lgamma(self.q1)\
        -(self.q1+1)*(torch.log(q4)-torch.digamma(q3)) - self.q2*q3/q4
        if self.prior_sparsity:
            # expectation of log pi
            expect_lpi = torch.digamma(a3)-torch.digamma(a3 + a4)
            # expectation of log(1-pi)
            expect_l1_pi = torch.digamma(a4)-torch.digamma(a3 + a4)
            # first part 
            lh_1 = torch.sum(expect_lpi*pi_local\
            + expect_l1_pi*(1. - pi_local)) \
            - self.p*0.5*self.beta_log_var_prior \
            - 0.5*(torch.sum(beta_var)+torch.sum(torch.square(self.beta_mu)))/beta_var_prior
            # expected log likelihood of global pi part
            lh_global_pi = (self.a1-1)*expect_lpi+(self.a2 - 1)* expect_l1_pi - torch.lgamma(self.a1)-torch.lgamma(self.a2)+torch.lgamma(self.a1+self.a2)
            return lh_1+lh_global_pi+lh_noise_var
        else:
            lh1 = torch.sum(torch.log(pi_global)*pi_local\
            + torch.log(1.-pi_global)*(1. - pi_local)) \
            - self.p*0.5*self.beta_log_var_prior \
            - 0.5*(torch.sum(beta_var)+torch.sum(torch.square(self.beta_mu)))/beta_var_prior
            return lh1+lh_noise_var
    
    def log_entropy(self, pi_local, a3, a4, q3, q4, pi_global):
        '''Calculate the entropy for the approximation posterior
        
        Parameters:
        ------------------
        pi_local: sparsity parameter for approximation posterior: p by 1
        a3: global sparsity beta distribution parameter for approximation posterior
        a4: global sparsity beta distribution parameter for approximation posterior
        q3: noise variance inverse gamma for approximation posterior
        q4: noise variance inverse gamma for approximation posterior
        pi_global: global sparsity if prior_sparsity=False, or else it is not used
        
        Return:
        -------------------
        The entropy for the approximation posterior
        '''
        entropy1 = -torch.sum(
            pi_local*torch.log(pi_local)-0.5*self.beta_log_var + (1-pi_local)*torch.log(1-pi_local)
        )
        if self.prior_sparsity:
            entropy_global_pi = torch.lgamma(a3)+torch.lgamma(a4) - torch.lgamma(a3+a4) - (a3 - 1)*torch.digamma(a3)\
                                - (a4 - 1)*torch.digamma(a4)+(a3+a4-2)*torch.digamma(a3+a4)
        else:
            entropy_global_pi = 0
        entropy_noise_var = q3+torch.log(q4)+torch.lgamma(q3)-(q3+1)*torch.digamma(q3)
        return entropy1+entropy_global_pi+entropy_noise_var
    
    def ELBO(self,X, y):
        '''
        Caculate the Evidence Lower Bound
        '''
        # get the current parameter after transformation
        noise_var, beta_var, pi_local,beta_var_prior,a3, a4, q3, q4,pi_global = self.get_para_orig_scale()
        # reparameterization
        #import pdb; pdb.set_trace()
        beta = self.beta_mu + torch.sqrt(beta_var)*torch.randn((self.n_E,self.p))
        # Gumbel-softmax sampling
        delta = nn.functional.gumbel_softmax(torch.stack( [ self.logit_pi_local.expand(self.n_E ,-1), -self.logit_pi_local.expand(self.n_E,-1) ], dim = 2 ),dim = 2, tau = self.tau, hard = self.hard)[:,:,0]
        # ELBO
        ELBO = self.log_data_lh(beta, delta, X, y, q3, q4) + \
            self.log_prior_expect_lh(pi_local, beta_var, beta_var_prior, a3, a4, q3, q4,pi_global) + \
            self.log_entropy(pi_local, a3, a4, q3, q4, pi_global)
        return ELBO
    
    def inference(self, X, num_samples = 500, plot = False, true_beta = None):
        '''Obtain the posterior and coefficient plot
        
        Parameters:
        --------------
        X: the data design matrix: n by p
        num_samples: number of samples to draw from the posterior
        plot: whether to plot the coefficient value
        true_beta: the coefficient to compare with
        
        Return:
        --------------
        A dictionary contains the posterior estimates (lower and upper band)
        A plot for the coefficents
        '''
        beta_mean = (self.beta_mu.detach()).numpy()
        beta_std = torch.exp(self.beta_log_var).detach().numpy()
        pi_local = torch.sigmoid(self.logit_pi_local.detach()).numpy()
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
        if self.prior_sparsity:
            # global pi posterior parameters
            a3 = torch.exp(self.log_a3).detach().numpy()
            a4 = torch.exp(self.log_a4).detach().numpy()
            global_pi_poster = np.random.beta(a3,a4, size = (num_samples,))
            global_pi_est = np.mean(global_pi_poster)
            global_pi_upper = np.quantile(global_pi_poster, q = 0.975)
            global_pi_lower = np.quantile(global_pi_poster, q = 0.025)
        else:
            global_pi_est = torch.sigmoid(self.logit_pi_global).detach().numpy()
            global_pi_upper = global_pi_est
            global_pi_lower = global_pi_est
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
        