import sys
sys.path.insert(1,'../functions/')
import torch
from torch import nn
import numpy as np
import data_sim
import matplotlib.pyplot as plt
import xarray as xr
import copy
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import time
from datetime import date
from sklearn.model_selection import ParameterGrid
# custom functions
from full_slab_spike_model_constructor import *
from training_func import *

today = date.today()
date = today.strftime("%d%m%Y")
real_data_index = True
G_21 = xr.open_dataset("../dataset/G_21_subset_imputed.nc")
G_21 = G_21.to_array().values.squeeze(0)
device = 'cuda'

file_name = '../results/sim_results'+date+'.csv'
if __name__ ==  '__main__': 
    df_total = pd.DataFrame()
    n_sim = 2
    h_v = [0.5,0.8, 0.2]
    n_v = [3565]
    p_v = [102484]
    p_causal_v = [300]
    t_v = [100] #number of moving averages
    patience_v = [50]# patience
    batch_size_v =[512]
    lr_general_v = [0.05]
    lr_pi_v = [0.5]
    gumbel_softmax_temp_v = [0.1, 0.5,1]
    gumbel_softmax_hard_v = [True, False]
    param_grid = {'h':h_v, 'n': n_v, 'p': p_v, 'p_causal': p_causal_v , 't': t_v, 
                  'patience':patience_v, 'batch_size':batch_size_v,
                  'lr_general':lr_general_v, 'lr_pi':lr_pi_v, 'gumbel_softmax_temp':gumbel_softmax_temp_v, 
                  'gumbel_softmax_hard':gumbel_softmax_hard_v}
    param_grid = ParameterGrid(param_grid)
    start = time.time()
    for param in param_grid:
        h = param['h']
        n = param['n']
        p = param['p']
        p_causal = param['p_causal']
        t = param['t']
        patience = param['patience']
        batch_size = param['batch_size']
        r_batch =  batch_size/n
        lr_general = param['lr_general']
        lr_pi = param['lr_pi']
        gumbel_softmax_temp = param['gumbel_softmax_temp']
        gumbel_softmax_hard = param['gumbel_softmax_hard']
        df_result = []
        net_parameters = 'h='+str(h)+'_n='+str(n)+'_p='+str(p)+'_p_causal='+str(p_causal)+'_t='+str(t)+"_patience="+str(patience)+"_batch_size="+str(batch_size)+\
                        '_lr_general='+str(lr_general)+'_lr_pi='+str(lr_pi)+'_gumbel_softmax_temp='+str(gumbel_softmax_temp)+'_gumbel_softmax_hard='+str(gumbel_softmax_hard)
        for i in range(n_sim):
            # permutate real data
            np.random.seed(seed = None)
            p_max = G_21.shape[1]
            n_max = G_21.shape[0]
            SNP_index = np.random.choice(p_max,size = p, replace = False)
            subject_index =  np.random.choice(n_max,size = n, replace = False)
            G_21_sub = G_21[np.ix_(subject_index, SNP_index)]
            freq_snp = G_21_sub.mean(axis = 0)/2
            sd_snp = np.sqrt(2*freq_snp*(1-freq_snp))
            G_21_sub = (G_21_sub-2*freq_snp)/sd_snp
            Xs = [G_21_sub]
            # Simulate data
            sim_class = data_sim.sim_tobit_data(n = n, p = p, p_causal = p_causal, rho = 0.1, var = 1, n_matrix = 1,h = h, bias = 1, Xs = Xs, scale_lambda =None)
            z, X, Xs, latent_mean, var_genetic, var_total, true_beta, y_star = sim_class.gen_data(seed = None)
            sim_data = Sim_Dataset(X.double().to(device),y_star.double().to(device))
            sim_data_loader = DataLoader(sim_data, batch_size = batch_size)
            model = linear_slab_spike(p = p, init_pi_local = 0.5, init_pi_global = 0.1, init_beta_var = 2, init_noise_var = 0.1,
                                     gumbel_softmax_temp = gumbel_softmax_temp, gumbel_softmax_hard = gumbel_softmax_hard, 
                                      a1= 0.1, a2=0.1, init_a3= 0.1, init_a4 = 0.1,
                                      q1 = 1.1, q2 = 1.1, init_q3 = 1.1, init_q4 = 1.1, n_E = 1
                                      , prior_sparsity = True, device = device
                                     ).double().to(device)
            optimizer = torch.optim.Adam(
                [{'params': model.beta_mu},
                 {'params': model.beta_log_var},
                 {'params': model.logit_pi_local, 'lr': lr_pi},
                 {'params': model.log_a3},
                 {'params': model.log_a4},
                 {'params': model.log_q3},
                 {'params': model.log_q4},
                 {'params': model.bias},
                 {'params': model.logit_pi_global, 'lr': lr_pi},
                 {'params': model.beta_log_var_prior},
                 {'params': model.log_var_noise},
                ],
                lr = lr_general)
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)
            _, result_dict = train_and_infer(model = model, optimizer = optimizer, sim_data_loader = sim_data_loader, lr_scheduler = lr_scheduler, t = t, patience = patience, X = X, plot = False, true_beta = true_beta)
            pd_cur = pd.DataFrame.from_dict(result_dict, orient='columns')
            df_result.append(pd_cur)
            df_ = pd.concat(df_result)
            torch.cuda.empty_cache()
            print('finish', end = ' ')
        df_ = pd.concat([pd.DataFrame([net_parameters]*df_.shape[0]), df_.reset_index()], axis = 1)
        df_total = pd.concat([df_total,df_])
        df_total.to_csv(file_name, index = False)
    end = time.time()
    print('Runtime of the program is ' + str(end -start) + ' seconds')