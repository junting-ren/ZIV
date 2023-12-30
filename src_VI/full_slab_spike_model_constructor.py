import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import math



class linear_slab_spike(nn.Module):
    def __init__(self, p, n_total, p_confound = 0,init_pi_local_max = 0.1, init_pi_local_min = 0.0, init_pi_global = 0.5, init_beta_var = 1, init_noise_var = 1,
                gumbel_softmax_temp = 0.5, gumbel_softmax_hard = False, a1= 1.1,a2=3.1, init_a3= 1.1, init_a4 = 5.1,
                b1 = 1.1, b2 = 1.1, init_b3 = 1.1, init_b4 = 1.1, n_E = 50, prior_sparsity = True,
                 prior_sparsity_beta = False, exact_lh = True, tobit = True, device = 'cpu'):
        '''Initialize fast variational inference Bayesian slab and spike linear model
        
        Parameters:
        ------------------------------------------------------------------------------------------
        p: number of features that needs variable selection, not including the bias
        n_total: total number of observations in the full dataset, for correcting the batch likelihood.
        p_confound: number of confounder features that does not need variable selection
        init_pi_local_max: the upper bound of uniform initilization for the non-null probability of each feature
        init_pi_local_min: the lower bound of uniform initilization for the non-null probability of each feature
        init_pi_global:  the initial value of global non-null probability, not used when prior_sparsity is True
        init_beta_var: the initial value of feature coefficent variance
        init_noise_var: inital value for noise variance
        gumbel_softmax_temp: Depreciated, gumbel softmax temperature value, the smaller the value, the more closer to the true binary sample but with greater graident variance
        gumbel_softmax_hard: Depreciated, if True, the returned samples will be discretized as one-hot vectors, but will be differentiated as if it is the soft sample in autograd
        a1: Depreciated (uniform prior now), prior beta distribution parameters for global sparsity
        a2: Depreciated (uniform prior now), prior beta distribution parameters for global sparsity
        init_a3: posterior beta distribution parameter 1
        init_a4:posterior beta distribution parameter 2
        b1: Depreciated (defaults to uniform prior now), prior inverse gamma distribution parameters for noise varaiance
        b2: Depreciated (defaults to uniform prior now), prior inverse gamma distribution parameters for noise varaiance
        init_b3: initial mean for log-normal posterior for noise variance
        init_b4: initial variance for log-normal posterior for noise variance
        n_E: Depreciated (defaults to exact likelihood calculation), number of samples for the empirical expectation of the data likelihood
        prior_sparsity: Defaults to True. If True, the global sparsity will have a prior or else it will be estimated through Maximum likelihood.
        prior_sparsity_beta: Defaults to False. If True, the global sparsity will have a beta prior or else it will be Uniform.
        exact_lh: Defaults to True. If True, the likelihood will be calculated exactly, else it will be estimated through Monte Carlo sampling.
        tobit: Defaults to True. Whether the outcome is censored or not.
        device: Defaults to 'cpu'. The device to run the model on. Other options include 'cuda' for GPU.
        '''
        super(linear_slab_spike, self).__init__()
        # Fixed values in the model
        self.device = device
        self.p = p #number of features
        self.p_confound = p_confound
        self.n_total = n_total
        self.tobit = tobit
        prior_uni = np.sqrt(6/p)/p # initial values for coefficient mean 
        self.a1 = torch.tensor((a1,)).to(device)# prior parameters for global pi
        self.a2 = torch.tensor((a2,)).to(device)# prior parameters for global pi
        self.b1 = torch.tensor((b1,)).to(device)#Priors parameters for noise variance
        self.b2 = torch.tensor((b2,)).to(device)#Priors parameters for noise variance
        self.tau = gumbel_softmax_temp# gumbel hyper parameters        
        self.hard = gumbel_softmax_hard# gumbel hyper parameters
        self.n_E = n_E #number of samples for empirical integration
        self.prior_sparsity = prior_sparsity # If True, the global sparsity will have a prior or else it will be estimated through Maximum likelihood.
        self.prior_sparsity_beta = prior_sparsity_beta
        # Variational approximation posterior distribution parameters
        self.beta_mu = nn.Parameter(torch.FloatTensor(size = (p,)).uniform_(-prior_uni,prior_uni)) # beta mean:Xavier Uniform Distribution initilization
        self.beta_log_var = nn.Parameter(torch.log(torch.rand((p,)))) # beta log variance: uniform 0,1 initilizaiton
        #self.logit_pi_local = nn.Parameter(torch.logit(torch.FloatTensor(size = (p,)).uniform_(init_pi_local-0.05,init_pi_local+0.05))) # sparsity for each feature, uniform initilized
        self.logit_pi_local = nn.Parameter(torch.logit(torch.FloatTensor(size = (p,)).uniform_(init_pi_local_min,init_pi_local_max))) # sparsity for each feature, uniform initilized
        self.log_a3 = nn.Parameter(torch.log(torch.tensor((init_a3,)))) # prior for global pi beta distribution
        self.log_a4 =  nn.Parameter(torch.log(torch.tensor((init_a4,))))# prior for global pi beta distribution
        self.b3 = nn.Parameter(torch.tensor((init_b3,)))# mean parameter for noise variance's log-normal approximation distribution
        self.log_b4 = nn.Parameter(torch.log(torch.tensor((init_b4,))))# log variance parameter for noise variance's log-normal approximation distribution
        # MLE parameters
        if self.p_confound>0:
            self.beta_confound = nn.Parameter(torch.FloatTensor(size = (p_confound,)).uniform_(-1,1))
        self.bias = nn.Parameter(torch.tensor((1.,)))# Bias
        self.logit_pi_global = nn.Parameter(torch.logit(torch.tensor(init_pi_global))) # global pi on logit scale 
        self.beta_log_var_prior = nn.Parameter(torch.log(torch.tensor(init_beta_var))) # beta prior log variance
        self.log_var_noise = nn.Parameter(torch.log(torch.tensor(init_noise_var))) # linear noise prior variance
        self.exact_lh = exact_lh
        
    def get_para_orig_scale(self):
        '''Function to calculate and return the parameters in the original scale
        '''
        return torch.exp(self.log_var_noise),torch.exp(self.beta_log_var), torch.sigmoid(self.logit_pi_local), \
                torch.exp(self.beta_log_var_prior), \
                torch.exp(self.log_a3), torch.exp(self.log_a4),\
                self.b3,torch.exp(self.log_b4), torch.sigmoid(self.logit_pi_global)
    
    def log_data_lh(self, beta, delta, X, y, b3, b4, beta_var):
        '''Calculate the expected data likelihood over the approximation posterior
        
        Parameters:
        ----------------------
        beta: sampled coefficient, p dimensional vector
        delta: sampled sparsity, p dimensional vector
        X: feature matrix, n by (p_confound+p) matrix
        y: outcome vector, n dimensional vector
        b3: mean parameter for log-normal approximation posterior for noise variance
        b4: variance parameter for log-normal approximation posterior for noise variance
        beta_var: beta variance vector for approximation posterior, p by 1

        Return:
        ----------------------
        Expected data likelihood over the approximation posterior
        '''
        n = X.shape[0]
        # X matrix should be n by self.p_confound+self.p  dimension
        if self.p_confound>0:
            X_conf = X[:,:self.p_confound]
            X = X[:,self.p_confound:]
            est_mean = self.beta_confound @ X_conf.t()+(beta*delta) @ X.t()+self.bias
        else:
            est_mean = (beta*delta) @ X.t()+self.bias
        if self.tobit:
            index0 = y==0
            index1 = y!=0
            n0 = np.sum(index0.cpu().detach().numpy())
            n1 = np.sum(index1.cpu().detach().numpy())
            if n0>0:
                X0 = X[index0,:]
                y0 = y[index0]
                est_mean0 = est_mean[index0]
            if n1>0:
                X1 = X[index1,:]
                y1 = y[index1]
                est_mean1 = est_mean[index1]
            if self.exact_lh and n1>0:
                mu_2 = beta**2
                pi_2 = delta**2
                diff = X1**2 @ (-mu_2*pi_2+(mu_2+beta_var)*delta)
                #diff = 0
                sum_squares = torch.square(y1-est_mean1) + diff
                l1 = -n1*0.5*b3-1/(2)*torch.exp(-b3+b4/2)*torch.sum(sum_squares)
            elif n1>0:
                l1 = torch.mean(-n1*0.5*b3-1/(2)*torch.exp(-b3+b4/2)*torch.sum(torch.square(y1-est_mean1)))
            else:
                l1 = 0
            if n0>0:
                #sigma = torch.sqrt(torch.distributions.gamma.Gamma(b3, b4).rsample())
                #l2 = torch.sum(torch.special.log_ndtr(-est_mean0/torch.sqrt(b4/(b3+1))))#plug in the mode for the variance
                value = -est_mean0
                scale = torch.sqrt(torch.exp(b3-b4))
                log_l = torch.log(torch.clamp(0.5 * (1 + torch.erf((value) * scale.reciprocal() / math.sqrt(2))), min = 1e-7))
                #import pdb; pdb.set_trace()
                #log_l[torch.isinf(log_l)].clamp(1e-7)
                l2 = torch.sum(log_l)
                # if torch.isinf(l2):
                #     import pdb; pdb.set_trace()
            else:
                l2 = 0
            return l1+l2
        else:
            if self.exact_lh:
                mu_2 = beta**2
                pi_2 = delta**2
                diff = X**2 @ (-mu_2*pi_2+(mu_2+beta_var)*delta)
                #diff = 0
                sum_squares = torch.square(y-est_mean) + diff
                return -n*0.5*b3-1/(2)*torch.exp(-b3+b4/2)*torch.sum(sum_squares)
                #return -n*0.5*(torch.log(b4)-torch.digamma(b3))-1/(2)*b3/b4*torch.sum(sum_squares)
            else:
                return torch.mean(-n*0.5*(torch.log(b4)-torch.digamma(b3))-1/(2)*b3/b4*torch.sum(torch.square(y-est_mean)))
    
    def log_prior_expect_lh(self, pi_local, beta_var, beta_var_prior, a3, a4, b3, b4, pi_global):
        '''Calculate the expected data likelihood over the approximation posterior
        
        Parameters:
        ------------------
        pi_local: non-null proportion parameter for approximating the posterior: p dimensional vector
        beta_var: variance for coefficient for approximating the posterior: p dimensional vector
        beta_var_prior: fixed prior variance for coefficient
        a3: global non-null proportion beta distribution parameter for approximating the posterior
        a4: global non-null proportion beta distribution parameter for approximating the posterior
        b3: mean parameter for log-normal approximation posterior for noise variance
        b4: variance parameter for log-normal approximation posterior for noise variance
        pi_global: Depreciated, global non-null proportion if prior_sparsity=False, or else it is not used
        
        Return:
        -------------------
        Expected prior likelihood over the approximation posterior
        '''
        lh_noise_var = 0
        if self.prior_sparsity:
            # expectation of log pi
            expect_lpi = torch.digamma(a3)-torch.digamma(a3 + a4)
            # expectation of log(1-pi)
            expect_l1_pi = torch.digamma(a4)-torch.digamma(a3 + a4)
            # first part 
            lh_1 = torch.sum(expect_lpi*pi_local\
            + expect_l1_pi*(1. - pi_local)) \
            - self.p*0.5*self.beta_log_var_prior \
            - 0.5*(torch.sum(beta_var*pi_local)+torch.sum(torch.square(self.beta_mu)*pi_local)+torch.sum((1-pi_local)*beta_var_prior))/beta_var_prior
            # expected log likelihood of global pi part
            if self.prior_sparsity_beta:
                lh_global_pi = (self.a1-1)*expect_lpi+(self.a2 - 1)* expect_l1_pi - torch.lgamma(self.a1)-torch.lgamma(self.a2)+torch.lgamma(self.a1+self.a2)
            else:
                lh_global_pi = 0 # since log(1)=0 for uniform
            return lh_1+lh_global_pi+lh_noise_var
        else:
            lh_1 = - self.p*0.5*self.beta_log_var_prior \
            - 0.5*(torch.sum(beta_var*pi_local)+torch.sum(torch.square(self.beta_mu)*pi_local)+torch.sum((1-pi_local)*beta_var_prior))/beta_var_prior
            return lh_1+lh_noise_var
    
    def log_entropy(self, pi_local, a3, a4, b3, b4):
        '''Calculate the entropy for the approximation posterior
        
        Parameters:
        ------------------------
        pi_local: sparsity parameter for approximation posterior: p by 1
        a3: global non-null proportion beta distribution parameter for approximation posterior
        a4: global non-null proportion beta distribution parameter for approximation posterior
        b3: mean parameter for log-normal approximation posterior for noise variance
        b4: variance parameter for log-normal approximation posterior for noise variance

        Return:
        -------------------
        The entropy for the approximation posterior
        '''
        entropy1 = -torch.sum(
            pi_local*torch.log(pi_local) + (1-pi_local)*torch.log(1-pi_local)-0.5*pi_local*self.beta_log_var-0.5*(1-pi_local)*self.beta_log_var_prior
        )
        if self.prior_sparsity:
            entropy_global_pi = torch.lgamma(a3)+torch.lgamma(a4) - torch.lgamma(a3+a4) - (a3 - 1)*torch.digamma(a3)\
                                - (a4 - 1)*torch.digamma(a4)+(a3+a4-2)*torch.digamma(a3+a4)
        else:
            entropy_global_pi = 0
        #entropy_noise_var = b3+torch.log(b4)+torch.lgamma(b3)-(b3+1)*torch.digamma(b3)
        entropy_noise_var = torch.log(torch.sqrt(b4)*torch.exp(b3+0.5))
        return entropy1+entropy_global_pi+entropy_noise_var
    
    def ELBO(self,X, y):
        '''
        Caculate the Evidence Lower Bound
        
        Parameters:
        ---------------------
        X: feature matrix, n by (p_confound+p) matrix
        y: outcome vector, n by 1
        
        Return:
        -------------------
        The evidence lower bound
        '''
        n_batch = X.shape[0]
        # get the current parameter after transformation
        noise_var, beta_var, pi_local,beta_var_prior,a3, a4, b3, b4,pi_global = self.get_para_orig_scale()
        # reparameterization
        #import pdb; pdb.set_trace()
        if self.exact_lh:
            beta = self.beta_mu
            delta = pi_local
        else:
            # Gumbel-softmax sampling
            delta = (nn.functional.gumbel_softmax(torch.stack( [ self.logit_pi_local.expand(self.n_E ,-1), -self.logit_pi_local.expand(self.n_E,-1) ], dim = 2 ),dim = 2, tau = self.tau, hard = self.hard)[:,:,0]).squeeze(0)
            beta = self.beta_mu*delta + torch.sqrt(beta_var*delta+beta_var_prior*(1-delta))*torch.randn((self.n_E,self.p), device = self.device).squeeze(0)

        # ELBO
        ELBO = self.log_data_lh(beta, delta, X, y, b3, b4, beta_var) + \
            n_batch/self.n_total*self.log_prior_expect_lh(pi_local, beta_var, beta_var_prior, a3, a4, b3, b4,pi_global) + \
            n_batch/self.n_total*self.log_entropy(pi_local, a3, a4, b3, b4)
        return ELBO
    
    def inference(self, est_mean, num_samples = 500, plot = False, true_beta = None):
        '''Obtain the posterior and coefficient plot
        
        Parameters:
        --------------
        est_mean: the estimate mean for X 
        num_samples: number of samples to draw from the posterior
        plot: whether to plot the coefficient value
        true_beta: the coefficient to compare with
        
        Return:
        --------------
        A dictionary contains the posterior estimates (lower and upper band)
        A plot for the coefficents
        '''
        beta_mean = (self.beta_mu.cpu().detach()).numpy()
        #beta_var = torch.exp(self.beta_log_var).cpu().detach().numpy()
        #beta_prior_var = torch.exp(self.beta_log_var_prior).cpu().detach().numpy()
        pi_local = torch.sigmoid(self.logit_pi_local.cpu().detach()).numpy()
        beta_plot = beta_mean*pi_local
        #import pdb; pdb.set_trace()
        #delta = np.random.binomial(n = 1, p = pi_local, size = (num_samples, self.p))
        #sample_beta = np.random.normal(loc = beta_mean*delta, scale = np.sqrt(beta_var*delta+beta_prior_var*(1-delta)), size = (num_samples, self.p))*delta # num_samples* p
        #est_mean = X.numpy() @ np.transpose(sample_beta) + self.bias.cpu().detach().numpy() # a n*num_samples matrix
        # Noise variance poseterior parameters
        b3 = self.b3.cpu().detach().numpy()
        b4 = np.exp(self.log_b4.cpu().detach().numpy())
        # Noise variance posterior
        noise_var_poster = np.exp(np.random.normal(b3,np.sqrt(b4), size = (num_samples,)))
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
            #global pi posterior parameters
            a3 = torch.exp(self.log_a3).cpu().detach().numpy()
            a4 = torch.exp(self.log_a4).cpu().detach().numpy()
            global_pi_poster = np.random.beta(a3,a4, size = (num_samples,))
            global_pi_est = np.mean(global_pi_poster)
            global_pi_upper_1 = np.quantile(global_pi_poster, q = 0.975)
            global_pi_lower_1 = np.quantile(global_pi_poster, q = 0.025)
            m = np.sum(np.random.binomial(n= 1, p = torch.sigmoid(self.logit_pi_local).detach().cpu().numpy(), size = (num_samples, self.p)), axis = 1)
            a3 = m+1
            a4 = self.p-m+1
            global_pi_poster = np.random.beta(a3,a4)
            #global_pi_est = np.mean(global_pi_poster)
            global_pi_upper_2 = np.quantile(global_pi_poster, q = 0.975)
            global_pi_lower_2 = np.quantile(global_pi_poster, q = 0.025)
            # combine
            global_pi_upper = max((global_pi_upper_1,global_pi_upper_2))
            global_pi_lower = min((global_pi_lower_1,global_pi_lower_2))
        else:
            m = np.sum(np.random.binomial(n = 1, p = torch.sigmoid(self.logit_pi_local).detach().cpu().numpy(), size = (num_samples, self.p)), axis = 1)
            a3 = m+1
            a4 = self.p-m+1
            global_pi_poster = np.random.beta(a3,a4)
            global_pi_est = np.mean(global_pi_poster)
            global_pi_upper = np.quantile(global_pi_poster, q = 0.975)
            global_pi_lower = np.quantile(global_pi_poster, q = 0.025)
        
        if true_beta is not None:
            #import pdb;pdb.set_trace()
            #Get the sensitivity and FDR using half of the estimated non-zero number
            n_est_positive_100 = int(global_pi_est*len(pi_local))
            n_est_positive_75 = int(global_pi_est*len(pi_local)*0.75)
            n_est_positive_50 = int(global_pi_est*len(pi_local)/2)
            n_est_positive_25 = int(global_pi_est*len(pi_local)/4)
            # p = len(pi_local)
            # pi_local_sorted = np.sort(pi_local)
            # sum_p = 0
            # for i in range(p):
            #     sum_p += 1 - pi_local_sorted[p-i-1]
            #     if sum_p/(i+1) > 0.05:
            #         n_est_positive = i
            #         break
            index_actual_positive = np.where(np.abs(true_beta)>0)[0]
            # 100% of the estimate PNN
            index_est_positive_100 = np.argsort(-np.abs(pi_local))[:n_est_positive_100]
            FDR_100 = np.mean(~np.isin(index_est_positive_100, index_actual_positive))
            sensitivity_100 = np.sum(np.isin(index_est_positive_100, index_actual_positive))/len(index_actual_positive) if len(index_actual_positive)>0 else 1
            # 75% of the estimate PNN
            index_est_positive_75 = np.argsort(-np.abs(pi_local))[:n_est_positive_75]
            FDR_75 = np.mean(~np.isin(index_est_positive_75, index_actual_positive))
            sensitivity_75 = np.sum(np.isin(index_est_positive_75, index_actual_positive))/len(index_actual_positive) if len(index_actual_positive)>0 else 1
            # 50% of the estimate PNN
            index_est_positive_50 = np.argsort(-np.abs(pi_local))[:n_est_positive_50]
            FDR_50 = np.mean(~np.isin(index_est_positive_50, index_actual_positive))
            sensitivity_50 = np.sum(np.isin(index_est_positive_50, index_actual_positive))/len(index_actual_positive) if len(index_actual_positive)>0 else 1
            # 25% of the estimate PNN
            index_est_positive_25 = np.argsort(-np.abs(pi_local))[:n_est_positive_25]
            FDR_25 = np.mean(~np.isin(index_est_positive_25, index_actual_positive))
            sensitivity_25 = np.sum(np.isin(index_est_positive_25, index_actual_positive))/len(index_actual_positive) if len(index_actual_positive)>0 else 1
        else:
            FDR_100 = None
            sensitivity_100 = None
            FDR_75 = None
            sensitivity_75 = None
            FDR_50 = None
            sensitivity_50 = None
            FDR_25 = None
            sensitivity_25 = None
        if plot and true_beta is not None:
            fig = plt.figure(figsize=(16,8), facecolor='white')
            ax = fig.add_subplot(1,1,1)
            ax.plot(np.arange(self.p), true_beta, \
                   linewidth = 3, color = "black", label = "ground truth")
            ax.scatter(np.arange(self.p), true_beta, \
                   s = 70, marker = '+', color = "black")
            ax.plot(np.arange(self.p),  beta_plot, \
                       linewidth = 3, color = "red", \
                       label = "ZIV estimates")
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
        return {'mean_h_est': [mean_h_est], 'h_est_upper': [upper], 'h_est_lower': [lower], 
                'mean_var_genetic': [var_genetic_mean], 'noise_var': [noise_var_est], 
                'global_pi':[global_pi_est], 'global_pi_upper':[global_pi_upper], 'global_pi_lower':[global_pi_lower],
                'global_pi_upper_1':[global_pi_upper_1], 'global_pi_lower_1':[global_pi_lower_1],
                'FDR_50':FDR_50, 'sensitivity_50':sensitivity_50, 'FDR_25':FDR_25, "sensitivity_25":sensitivity_25,
                'FDR_100':FDR_100, "sensitivity_100":sensitivity_100,'FDR_75':FDR_75, "sensitivity_100":sensitivity_75
               }
    
    def cal_mean_batch(self, X_batch, sample_beta, y = None):
        '''
        Calculate the mean prediction for the batch
        
        Parameters:
        -----------------------
        X_batch: current batch feature matrix, n_batch by (p_confound+p) matrix
        sample_beta: num_samples by p 
        y: current batch response vector, n_batch dimensional  vector

        Return:
        -----------------------
        A numpy 1D array containing the prediction for the batch
        '''
        if self.p_confound>0:
            est_mean = np.expand_dims(X_batch[:,:self.p_confound].cpu().detach().numpy() @ np.transpose(self.beta_confound.cpu().detach().numpy()), axis = 1) + X_batch[:,self.p_confound:].cpu().detach().numpy() @ np.transpose(sample_beta) + self.bias.cpu().detach().numpy() # a n_batch*num_samples matrix
        else:
            est_mean = X_batch.cpu().detach().numpy() @ np.transpose(sample_beta) + self.bias.cpu().detach().numpy() # a n_batch*num_samples matrix
        point_est = np.mean(est_mean, axis = 1)
        error = point_est*(point_est>0) - y.cpu().detach().numpy()
        return est_mean, error, point_est
    
    def predict(self, X):
        '''
        Predict the response for the input feature matrix
        
        Parameters:
        -----------------------
        X: current batch feature matrix

        Return:
        -----------------------
        A numpy 1D n_batch array prediction for the input feature matrix
        '''
        beta_mean = (self.beta_mu.cpu().detach()).numpy()
        pi_local = torch.sigmoid(self.logit_pi_local.cpu().detach()).numpy()
        sample_beta = beta_mean*pi_local
        #import pdb; pdb.set_trace()
        if self.p_confound>0:
            #import pdb; pdb.set_trace()
            est_mean = X[:,:self.p_confound].cpu().detach().numpy() @ np.transpose(self.beta_confound.cpu().detach().numpy()) + \
                X[:,self.p_confound:].cpu().detach().numpy() @ np.transpose(sample_beta) + self.bias.cpu().detach().numpy() # a n_batch*num_samples matrix
        else:
            est_mean = X.cpu().detach().numpy() @ np.transpose(sample_beta) + self.bias.cpu().detach().numpy() # a n_batch*num_samples matrix
        return est_mean
    
    def sample_beta(self, num_samples):
        '''Draw from the posterior distribution of beta

        Parameters:
        -----------------------
        num_samples: number of samples to draw

        Return:
        -----------------------
        a numpy array for the sample betas matrix num_samples by p 
        '''
        beta_mean = (self.beta_mu.cpu().detach()).numpy()
        beta_var = torch.exp(self.beta_log_var).cpu().detach().numpy()
        beta_prior_var = torch.exp(self.beta_log_var_prior).cpu().detach().numpy()
        pi_local = torch.sigmoid(self.logit_pi_local.cpu().detach()).numpy()
        #import pdb; pdb.set_trace()
        delta = np.random.binomial(n = 1, p = pi_local, size = (num_samples, self.p))
        sample_beta = np.random.normal(loc = beta_mean*delta, scale = np.sqrt(beta_var*delta+beta_prior_var*(1-delta)), size = (num_samples, self.p))*delta # num_samples* p
        return sample_beta
        