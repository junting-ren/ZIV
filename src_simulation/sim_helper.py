import sys
sys.path.insert(1,'../src_simulation/')
sys.path.insert(1,'../src_VI/')
sys.path.insert(1,'../src_MCMC/')
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import data_sim
from full_slab_spike_model_constructor import *
from training_func import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV,RidgeCV
import pyreadr
from sklearn.model_selection import ParameterGrid
import os

import time

from mcmc_linear import experiment, linear_mcmc_model, tobit_mcmc_model
import jax.numpy as jnp

def one_run(X, train_index, test_index, h, percent_causal, beta_var,rho, linear_outcome = False, compare_mcmc = False, n_sub = None, p_sub = None):
    #import pdb; pdb.set_trace()
    # Check if we need to simulate the data
    if not isinstance(X,np.ndarray):# If this is simulated data then X = "None", or else it would be ndarrayq
        p_causal = int(p_sub*percent_causal)
        sim_class = data_sim.sim_tobit_data(n = n_sub, p = p_sub, p_causal = p_causal,p_confound =0, rho = rho, var = 1,
                                            n_matrix = 1,h = h, bias = 0, Xs = None, scale_lambda =None, beta_var= beta_var)
        z, X, Xs, latent_mean, var_genetic, var_total, true_beta, y_star = sim_class.gen_data(seed = None)
    else:
        n = X.shape[0]
        p_confound = 0
        p = X.shape[1]-p_confound
        p_causal = int(p*percent_causal)
        # sub sample data
        if n_sub is not None:
            X = pd.DataFrame(X).sample(n = n_sub, axis = 0)
            X = np.array(X)
        if p_sub is not None:
            X = pd.DataFrame(X).sample(n = p_sub, axis = 1)
            X = np.array(X)
        # simulate data
        sim_class = data_sim.sim_tobit_data(n = n, p = p, p_causal = p_causal,p_confound =0, rho = None, var = None,
                                            n_matrix = 1,h = h, bias = 0, Xs = [X], scale_lambda =None, beta_var= beta_var)
        z, X, Xs, latent_mean, var_genetic, var_total, true_beta, y_star = sim_class.gen_data(seed = None)
    ##################################
    ### if we need linear outcome  
    ##################################
    if linear_outcome:
        z = y_star
    ##################################
    ### if we need linear outcome  
    ##################################
    #self.X_train, self.X_test, self.z_train, self.z_test = train_test_split(X, z, test_size=0.2, random_state=42)
    if n_sub is not None or p_sub is not None:
        n = X.shape[0]
        indices = np.random.permutation(n)
        split = int(n*0.8)
        train_index, test_index = indices[:split], indices[split:]
    X_train = X[train_index,:]
    X_test = X[test_index, :]
    z_train = z[train_index]
    z_test = z[test_index]
    batch_size = len(train_index)
    # Run the model
    device = 'cpu'
    # sim one data, where the true h, percent_causal and beta known
    # Run the model on the data and return 
    # the estimate and coverage for heritability, percentage causal, prediction bias in MAE, sensitivity and FDR
    n, p = X_train.shape
    sim_data = Sim_Dataset(torch.tensor(X_train).float(),torch.tensor(z_train).float(), device = 'cpu')
    sim_data_loader = DataLoader(sim_data, batch_size = batch_size)
    model = linear_slab_spike(p = p, n_total = n, p_confound = 0, init_pi_local_max = 0.2, 
                              init_pi_local_min = 0.0,init_pi_global = 0.5, init_beta_var =1, init_noise_var = 1,
                              gumbel_softmax_temp = 1, gumbel_softmax_hard = False, 
                              a1= 0.1, a2=0.1, init_a3= 1.1, init_a4 = 1.1,
                              b1 = 1.1, b2 = 1.1, init_b3 = 10.0, init_b4 = 0.1, n_E = 1
                              , prior_sparsity = True, prior_sparsity_beta = False,exact_lh = True,tobit = not linear_outcome, device = device
                             ).double().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.1)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.8)
    t = 100 #number of moving averages
    patience = 300# patience
    start = time.time()
    best_model, error, point_est, result_dict = train_and_infer(model = model, optimizer = optimizer, 
                                                                sim_data_loader = sim_data_loader, 
                                                                lr_scheduler = lr_scheduler, t = t, patience = patience,
                                                                X = X_train, plot = False, 
                                                                true_beta = true_beta, verbose = False)
    end = time.time()
    total_time_VI = end - start
    # latent model prediction
    z_pred = best_model.predict(torch.tensor(X_test).float())
    mae = np.mean(np.abs(z_pred*(z_pred>0) - z_test))
    #import pdb; pdb.set_trace()
    # Lasso
    reg = LassoCV(cv = 5, alphas = (0.001,0.01,0.1,1,10), max_iter=10000)
    reg.fit(X_train,z_train)
    z_pred = reg.predict(X_test)
    mae_lasso = np.mean(np.abs(z_pred*(z_pred>0)-z_test))
    # Lasso with only non-zero outcomes
    reg = LassoCV(cv = 5, alphas = (0.001,0.01,0.1,1,10), max_iter=10000)
    reg.fit(X_train[z_train>0,:],z_train[z_train>0])
    z_pred = reg.predict(X_test)
    mae_lasso_nonzero = np.mean(np.abs(z_pred*(z_pred>0)-z_test))
    # Ridge
    reg = RidgeCV(cv = 5, alphas = (0.001,0.01,0.1,1,10))
    reg.fit(X_train,z_train)
    z_pred = reg.predict(X_test)
    mae_ridge = np.mean(np.abs(z_pred*(z_pred>0)-z_test))
    # Ridge with only non-zero outcomes
    reg = RidgeCV(cv = 5, alphas = (0.001,0.01,0.1,1,10))
    reg.fit(X_train[z_train>0,:],z_train[z_train>0])
    z_pred = reg.predict(X_test)
    mae_ridge_nonzero = np.mean(np.abs(z_pred*(z_pred>0)-z_test))
    # This should return as a dictionary that contains the outcome and beta; 
    #the estimate and coverage for heritability, percentage causal, prediction bias in MAE, sensitivity and FDR
    result_dict['mae_latent'] = mae
    result_dict['mae_lasso'] = mae_lasso
    result_dict['mae_ridge'] = mae_ridge
    result_dict['mae_lasso_nonzero'] = mae_lasso_nonzero
    result_dict['mae_ridge_nonzero'] = mae_ridge_nonzero
    result_dict['n'] = n
    result_dict['p'] = p
    result_dict['linear_outcome'] = linear_outcome
    # MCMC model
    if compare_mcmc:
        start = time.time()
        if linear_outcome:
            exp = experiment(linear_mcmc_model, z_train, X_train)
        else:
            exp = experiment(tobit_mcmc_model, z_train, X_train)
        exp.train(step_size =1,verbose = False, num_samples = 1000, warmup_steps=1000)
        #import pdb; pdb.set_trace()
        posterior = exp.mcmc.get_samples()
        end = time.time()
        total_time_MCMC = end - start
        beta = posterior["beta"]* posterior["delta"]
        pi_local = np.mean(posterior["delta"], axis = 0)
        g_var = (beta @ jnp.transpose(X) ).var(axis = 1)
        dat_var = posterior["var_error"]
        heritability = g_var/(dat_var+g_var)
        mean_h_mcmc = np.mean(heritability)
        low_h_mcmc = np.quantile(heritability, 0.025)
        up_h_mcmc = np.quantile(heritability, 0.975)
        global_pi_mcmc = posterior['pi']
        mean_pi_mcmc = np.mean(global_pi_mcmc)
        low_pi_mcmc = np.quantile(global_pi_mcmc, 0.025)
        up_pi_mcmc = np.quantile(global_pi_mcmc, 0.975)
        result_dict['mean_h_mcmc'] = mean_h_mcmc
        result_dict['low_h_mcmc'] = low_h_mcmc
        result_dict['up_h_mcmc'] = up_h_mcmc
        result_dict['global_pi_mcmc'] = mean_pi_mcmc
        result_dict['low_global_pi_mcmc'] = low_pi_mcmc
        result_dict['up_global_pi_mcmc'] = up_pi_mcmc
        result_dict['total_time_VI'] = total_time_VI
        result_dict['total_time_MCMC'] = total_time_MCMC
        # Get sensitivity and FDR from MCMC
        n_est_positive_100 = int(mean_pi_mcmc*len(pi_local))
        n_est_positive_75 = int(mean_pi_mcmc*len(pi_local)*0.75)
        n_est_positive_50 = int(mean_pi_mcmc*len(pi_local)/2)
        n_est_positive_25 = int(mean_pi_mcmc*len(pi_local)/4)
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
        result_dict["FDR_100_mcmc"] = np.mean(~np.isin(index_est_positive_100, index_actual_positive))
        result_dict["sensitivity_100_mcmc"] = np.sum(np.isin(index_est_positive_100, index_actual_positive))/len(index_actual_positive) if len(index_actual_positive)>0 else 1
        # 75% of the estimate PNN
        index_est_positive_75 = np.argsort(-np.abs(pi_local))[:n_est_positive_75]
        result_dict["FDR_75_mcmc"] = np.mean(~np.isin(index_est_positive_75, index_actual_positive))
        result_dict["sensitivity_75_mcmc"] = np.sum(np.isin(index_est_positive_75, index_actual_positive))/len(index_actual_positive) if len(index_actual_positive)>0 else 1
        # 50% of the estimate PNN
        index_est_positive_50 = np.argsort(-np.abs(pi_local))[:n_est_positive_50]
        result_dict["FDR_50_mcmc"] = np.mean(~np.isin(index_est_positive_50, index_actual_positive))
        result_dict["sensitivity_50_mcmc"] = np.sum(np.isin(index_est_positive_50, index_actual_positive))/len(index_actual_positive) if len(index_actual_positive)>0 else 1
        # 25% of the estimate PNN
        index_est_positive_25 = np.argsort(-np.abs(pi_local))[:n_est_positive_25]
        result_dict["FDR_25_mcmc"] = np.mean(~np.isin(index_est_positive_25, index_actual_positive))
        result_dict["sensitivity_25_mcmc"] = np.sum(np.isin(index_est_positive_25, index_actual_positive))/len(index_actual_positive) if len(index_actual_positive)>0 else 1
    result_dict['true_h'] = h
    result_dict['true_pi'] = percent_causal
    result_dict['beta_var'] = beta_var
    result_dict['rho'] = rho
    return [z_train,z_test, pd.DataFrame(result_dict)]

def one_run_wrapper(kwargs):
    return one_run(**kwargs)


class sim_helper(object):
    def __init__(self, n_sim, heritability_l, percent_causal_l, beta_var_l, image_modality, linear_outcome_l = [False],
                 random_seed = 1, path = '', file_suffix = '',compare_mcmc = False, n_sub_l = None, p_sub_l = None, 
                 sim_data = False, rho_l = [None], data_path = './'):
        self.n_sim = n_sim
        self.heritability_l = heritability_l
        self.percent_causal_l = percent_causal_l
        self.beta_var_l = beta_var_l
        self.image_modality = image_modality
        self.random_seed = random_seed
        self.path = path
        self.compare_mcmc = compare_mcmc
        self.n_sub_l = n_sub_l
        self.p_sub_l = p_sub_l
        self.sim_data = sim_data
        self.rho_l = rho_l
        self.linear_outcome_l = linear_outcome_l
        self.file_suffix = file_suffix
        self.data_path = data_path
    def load_clean_data(self):
        # save the data into self
        # will use this multiple time
        #self.data_path = '/Users/juntingren/Desktop/L0_VI_Bayesian_full_experimentation_code/dataset/'
        ABCD = pd.read_csv(self.data_path+'abcd.csv')
        list_ROI = pyreadr.read_r(self.data_path+'ABCD_ROI.list.RData')
        # 'rsmri_list', 'tfmri_list', 'smri_list', 'dti_list', 'rsi_list'
        ABCD_sub = ABCD.loc[:,np.isin(ABCD.columns,list(np.squeeze(list_ROI[self.image_modality].values,1))+['subjectid', 'eventname', 'demo_rel_family_id.bl'])]
        ABCD_sub = ABCD_sub.loc[~ABCD_sub.isnull().any(axis = 1),:]
        ABCD_sub = ABCD_sub.groupby('demo_rel_family_id.bl', group_keys=False).apply(lambda x: x.sample(1, random_state = self.random_seed)).reset_index(drop=True)
        ABCD_sub = ABCD_sub.groupby(['subjectid',"eventname"], group_keys=False).apply(lambda x: x.sample(1, random_state = self.random_seed)).reset_index(drop=True)
        ABCD_sub = ABCD_sub.loc[:,np.isin(ABCD_sub.columns,list(np.squeeze(list_ROI[self.image_modality].values,1)))]
        scaler = StandardScaler()
        scaler.fit(ABCD_sub)
        ABCD_sub = scaler.transform(ABCD_sub)
        self.data = ABCD_sub
    def full_run(self):
        # loop over every parameter 
        # For every parameter run, we use parallel mapping function to run over the n_sim
        if self.sim_data:
            self.data = "None"
            train_index = "None"
            test_index = "None"
        else:
            self.load_clean_data()
            n = self.data.shape[0]
            indices = np.random.permutation(n)
            split = int(n*0.8)
            train_index, test_index = indices[:split], indices[split:]
            # Save the feature matrix
            data_save_train = pd.DataFrame(self.data[train_index,:])
            data_save_train.to_csv(os.path.join(self.path, self.image_modality+'_train_standardized_features.csv'), index = False)
            data_save_test = pd.DataFrame(self.data[test_index,:])
            data_save_test.to_csv(os.path.join(self.path, self.image_modality+'_test_standardized_features.csv'), index = False)
        #import pdb; pdb.set_trace()
        # setting up the parameter grid
        if self.n_sub_l is None:
            self.n_sub_l = [n]
        if self.p_sub_l is None:
            self.p_sub_l = [self.data.shape[1]]
        param_grid = {'X': [self.data], 'train_index': [train_index], 'test_index': [test_index],'h':self.heritability_l,
                      'percent_causal': self.percent_causal_l, 'beta_var': self.beta_var_l, 'compare_mcmc':[self.compare_mcmc],
                      'n_sub':self.n_sub_l,'p_sub':self.p_sub_l, 'rho':self.rho_l , 'linear_outcome':self.linear_outcome_l
                     }
        param_grid = ParameterGrid(param_grid)
        df_result_l = []
        z_train_l = []
        z_test_l = []
        for param in param_grid:
            ctx = torch.multiprocessing.get_context('spawn')
            pool_obj = ctx.Pool()
            # pool_obj = multiprocessing.Pool()
            cur_para = (self.n_sim*(param,))
            #import pdb; pdb.set_trace()
            #one_run_wrapper(param)
            result = pool_obj.map(one_run_wrapper, cur_para)
            pool_obj.close()
            pool_obj.join()
            df_result_l.extend([x[2] for x in result])
            save_result_name = 'result'+'_'+self.file_suffix+'.csv'
            #import pdb; pdb.set_trace()
            pd.concat(df_result_l).to_csv(os.path.join(self.path, save_result_name), index = False)
            z_train_l.extend([x[0] for x in result])
            z_test_l.extend([x[1] for x in result])
            pd.DataFrame(z_train_l).to_csv(os.path.join(self.path, 'z_train'+'_'+self.file_suffix+'.csv'), index = False)
            pd.DataFrame(z_test_l).to_csv(os.path.join(self.path, 'z_test'+'_'+self.file_suffix+'.csv'), index = False)
            # para_df = pd.DataFrame(np.repeat(df.values,df_.shape[0], axis=0))
            # para_df.columns = df.columns
            # df_ = pd.concat([para_df, df_], axis = 1)
            # df_total = pd.concat([df_total,df_])
            # df_total.to_csv(file_name, index = False)
    
        