import scipy
import matplotlib.pyplot as plt
#from tobit_dist import tobit_dist
import numpy as np
import torch
#import pyro.distributions as dist

class sim_tobit_data(object):
    def __init__(self, n, p, p_causal, rho, var, n_matrix, h, p_confound = 0, scale_lambda = None, Xs = None, beta = None, bias = 0, beta_var =1):
        """
        Initialize the parameters for data simulation process. This simulation process tries to generate different sub-feature matrices 
        where the features within one sub-matrix is correlated but independent with all other sub-matrices. 
        The model can run the sub-feature matrices and sum the heritability estimate to obtain overall heritability.
        
        Parameters:
        ---------------------
        n: number of total samples
        p: number of features for one sub-feature matrix
        p_causal: number of causal features for one sub-feature matrix
        rho: correlation parameter for AR1 correlation matrix
        var: variance for the correlation matrix assuming each feature has the same variance
        n_matrix: number of sub-feature matrix to generate
        h: the total heritability
        scale_lambda: default to None, if not None, we generate beta with a horse shoe prior , or else it is a 
        Xs: defaults to None. If not None, a list for sub-feature matrices (real SNP)
        beta: defaults to None. If not None, a vector of values for the betas
        bias: the bias term for the betas, defaults to 0
        """
        self.n = n
        self.p = p
        self.p_causal = p_causal
        self.p_confound = p_confound
        self.rho = rho
        self.var = var
        self.beta_var=beta_var
        self.n_matrix = n_matrix
        self.total_p = n_matrix*p
        self.total_p_causal = n_matrix*p_causal
        self.beta = beta
        self.scale_lambda = scale_lambda
        self.h = h
        self.bias = bias
        self.Xs = Xs
    def create_Xs(self, n, p, rho, var, n_matrix):
        exponential = np.abs(np.array(np.repeat(range(0,p),p)).reshape(p,p) - np.array(range(0,p)))
        cov = rho**exponential*var
        p = cov.shape[0]
        Xs_list = []
        for i in range(n_matrix):
            Xs_list.append(np.random.multivariate_normal(np.repeat(0,p), cov = cov, size = n))
        return Xs_list
    def create_beta(self, rng):
        # TODO: When Xs is not None, this is not taken care of in this version
        if self.scale_lambda is not None:
            beta = rng.standard_normal(size = self.total_p)
            lambda_ = dist.HalfCauchy(scale=self.scale_lambda).sample([self.total_p]).cpu().numpy()
            beta = beta*lambda_
        else:
            #import pdb; pdb.set_trace()
            SNP_c_index = np.random.choice(self.total_p ,size = self.total_p_causal, replace = False)
            beta = np.repeat(0.0,self.total_p)
            beta[SNP_c_index] = rng.standard_normal(size = self.total_p_causal)*self.beta_var**0.5
            beta_confound = rng.standard_normal(size = self.p_confound)
            beta =  np.concatenate([beta_confound, beta], axis = 0)
        return beta
    def gen_data(self,seed = None):
        """
        
        Returns
        --------------------
        A tuple consists of the following:
        z: the semi-positive outcomes, torch tensor
        X: the full feature matrix, torch tensor
        Xs: the list of sub-feature matrices torch tensors
        latent_mean: the mean on the latent variable
        y: the latent mean adding the noise
        """
        rng = np.random.default_rng(seed)
        if self.beta is None:
            self.beta = self.create_beta(rng)
        if self.Xs is None:
            Xs = self.create_Xs(self.n, self.p+self.p_confound, self.rho, self.var, self.n_matrix)
        else:
            Xs = self.Xs
        X = np.concatenate(Xs,axis = 1)
        #import pdb; pdb.set_trace()
        latent_mean = np.matmul(X, self.beta)+self.bias
        #import pdb;pdb.set_trace()
        var_genetic = np.mean(latent_mean**2) - np.mean(latent_mean)**2
        # std for the errors
        sigma_e = np.sqrt((1-self.h)*var_genetic/self.h)
        var_total = sigma_e**2+var_genetic
        #z = tobit_dist(mean = torch.tensor(latent_mean), sigma = torch.tensor(sigma_e)).sample()
        y_star = latent_mean + rng.standard_normal(size = X.shape[0])*sigma_e
        #import pdb;pdb.set_trace()
        z = np.where(y_star <= 0, 0,y_star)
        return z, X, Xs, latent_mean, var_genetic, var_total, self.beta, y_star
    
def show_sim(z, latent_mean, var_total):
    figure, axis = plt.subplots(1, 2)
    axis[0].hist(z, bins='auto')
    axis[0].set_title("Zero-inflated outcome")
    axis[1].hist(latent_mean, bins='auto')
    axis[1].set_title("Latent mean")
    plt.show()
    var_genetic = np.mean(latent_mean**2) - np.mean(latent_mean)**2
    h = var_genetic/var_total
    print('------------------------------------')
    print('genetic variance is ' + str(var_genetic))
    print('total variance is'+ str(var_total))
    print('heritability is ' + str(h))
    print('------------------------------------')