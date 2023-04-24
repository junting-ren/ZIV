import sys
sys.path.insert(1,'../functions/')
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


def one_run(X, train_index, test_index, h, percent_causal, beta_var):
    
    # simulate data
    batch_size = len(train_index)
    n = X.shape[0]
    p_confound = 0
    p = X.shape[1]-p_confound
    p_causal = int(p*percent_causal)
    sim_class = data_sim.sim_tobit_data(n = n, p = p, p_causal = p_causal,p_confound =0, rho = None, var = None,
                                        n_matrix = 1,h = h, bias = 0, Xs = [X], scale_lambda =None, beta_var= beta_var)
    z, X, Xs, latent_mean, var_genetic, var_total, true_beta, y_star = sim_class.gen_data(seed = None)
    #self.X_train, self.X_test, self.z_train, self.z_test = train_test_split(X, z, test_size=0.2, random_state=42)
    X_train = X[train_index,:]
    X_test = X[test_index, :]
    z_train = z[train_index]
    z_test = z[test_index]
    
    # Run the model
    device = 'cpu'
    # sim one data, where the true h, percent_causal and beta known
    # Run the model on the data and return 
    # the estimate and coverage for heritability, percentage causal, prediction bias in MAE, sensitivity and FDR
    n, p = X_train.shape
    sim_data = Sim_Dataset(torch.tensor(X_train).float(),torch.tensor(z_train).float(), device = 'cpu')
    sim_data_loader = DataLoader(sim_data, batch_size = batch_size)
    model = linear_slab_spike(p = p, n_total = n, p_confound = 0, init_pi_local_max = 0.05, 
                              init_pi_local_min = 0.0,init_pi_global = 0.5, init_beta_var =1, init_noise_var = 1,
                              gumbel_softmax_temp = 1, gumbel_softmax_hard = False, 
                              a1= 0.1, a2=0.1, init_a3= 1.1, init_a4 = 1.1,
                              b1 = 1.1, b2 = 1.1, init_b3 = 10.0, init_b4 = 0.1, n_E = 1
                              , prior_sparsity = True, prior_sparsity_beta = False,exact_lh = True,tobit = True, device = device
                             ).double().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.1)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.8)
    t = 100 #number of moving averages
    patience = 100# patience
    best_model, error, point_est, result_dict = train_and_infer(model = model, optimizer = optimizer, 
                                                                sim_data_loader = sim_data_loader, 
                                                                lr_scheduler = lr_scheduler, t = t, patience = patience,
                                                                X = X_train, plot = False, 
                                                                true_beta = true_beta, verbose = False)
    # latent model prediction
    z_pred = best_model.predict(torch.tensor(X_test).float())
    mae = np.mean(np.abs(z_pred*(z_pred>0) - z_test))
    # lasso and ridge prediction
    reg = LassoCV(cv = 5, alphas = (0.001,0.01,0.1,1,10), max_iter=10000)
    reg.fit(X_train,z_train)
    z_pred = reg.predict(X_test)
    mae_lasso = np.mean(np.abs(z_pred*(z_pred>0)-z_test))
    # Ridge
    reg = RidgeCV(cv = 5, alphas = (0.001,0.01,0.1,1,10))
    reg.fit(X_train,z_train)
    z_pred = reg.predict(X_test)
    mae_ridge = np.mean(np.abs(z_pred*(z_pred>0)-z_test))
    # This should return as a dictionary that contains the outcome and beta; 
    #the estimate and coverage for heritability, percentage causal, prediction bias in MAE, sensitivity and FDR
    result_dict['mae_latent'] = mae
    result_dict['mae_lasso'] = mae_lasso
    result_dict['mae_ridge'] = mae_ridge
    result_dict['true_h'] = h
    result_dict['true_pi'] = percent_causal
    result_dict['beta_var'] = beta_var
    return [z_train,z_test, pd.DataFrame(result_dict)]

def one_run_wrapper(kwargs):
    return one_run(**kwargs)


class sim_helper(object):
    def __init__(self, n_sim, heritability_l, percent_causal_l, beta_var_l, image_modality, random_seed = 1, path = ''):
        self.n_sim = n_sim
        self.heritability_l = heritability_l
        self.percent_causal_l = percent_causal_l
        self.beta_var_l = beta_var_l
        self.image_modality = image_modality
        self.random_seed = random_seed
        self.path = path
        
    def load_clean_data(self):
        # save the data into self
        # will use this multiple time
        ABCD = pd.read_csv('../dataset/abcd.csv')
        list_ROI = pyreadr.read_r('../dataset/ABCD_ROI.list.RData')
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
        self.load_clean_data()
        # loop over every parameter 
        # For every parameter run, we use parallel mapping function to run over the n_sim
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
        df_result = []
        param_grid = {'X': [self.data], 'train_index': [train_index], 'test_index': [test_index],'h':self.heritability_l,
                      'percent_causal': self.percent_causal_l, 'beta_var': self.beta_var_l
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
            result = pool_obj.map(one_run_wrapper, cur_para)
            pool_obj.close()
            pool_obj.join()
            df_result_l.extend([x[2] for x in result])
            pd.concat(df_result_l).to_csv(os.path.join(self.path, 'result.csv'), index = False)
            z_train_l.extend([x[0] for x in result])
            z_test_l.extend([x[1] for x in result])
            pd.DataFrame(z_train_l).to_csv(os.path.join(self.path, 'z_train.csv'), index = False)
            pd.DataFrame(z_test_l).to_csv(os.path.join(self.path, 'z_test.csv'), index = False)
            # para_df = pd.DataFrame(np.repeat(df.values,df_.shape[0], axis=0))
            # para_df.columns = df.columns
            # df_ = pd.concat([para_df, df_], axis = 1)
            # df_total = pd.concat([df_total,df_])
            # df_total.to_csv(file_name, index = False)    
    
        