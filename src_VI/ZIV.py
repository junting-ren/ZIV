from torch.utils.data import DataLoader
import sys
sys.path.insert(1,'../src_VI/')
import torch
import numpy as np
import pandas as pd
from full_slab_spike_model_constructor import *
from training_func import *
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class ZIV(object):
    def __init__(self, data, outcome_name, 
                 feature_conti_name, feature_cate_name,
                 confounder_conti_name, confounder_cate_name,
                 batch_size = None, 
                 device = 'cpu'):
        '''Initialize the ZIV model and dataset
        
        Parameters
        ----------
        data : pandas dataframe, n by p
        outcome_name : string, the name of the outcome
        feature_conti_name : list of strings or empty list, the name of the continuous features
        feature_cate_name : list of strings or empty list, the name of the categorical features
        confounder_conti_name : list of strings or empty list, the name of the continuous confounders
        confounder_cate_name : list of strings or empty list, the name of the categorical confounders
        batch_size: int, batch size, default None (full batch)
        device: string, 'cpu' or 'cuda'

        Returns
        -------
        None.
        '''
        self.outcome_name = outcome_name
        self.feature_conti_name = feature_conti_name
        self.feature_cate_name = feature_cate_name
        self.confounder_conti_name = confounder_conti_name
        self.confounder_cate_name = confounder_cate_name
        self.batch_size = batch_size
        self.device = device

        # Onehot Encode the categorical variables
        if len(self.confounder_cate_name) >0:
            self.OHE_conf = OneHotEncoder(drop = 'first',sparse_output = False).fit(data[self.confounder_cate_name])
            encoded_cols_conf = list(self.OHE_conf.get_feature_names_out(self.confounder_cate_name))
            data[encoded_cols_conf] = OHE_conf.transform(data[self.confounder_cate_name])
        else:
            encoded_cols_conf = []
        if len(self.feature_cate_name)>0:
            self.OHE_feature = OneHotEncoder(drop = 'first',sparse_output = False).fit(data[self.feature_cate_name])
            encoded_cols_feature = list(self.OHE_feature.get_feature_names_out(self.feature_cate_name))
            data[encoded_cols_feature] = OHE_feature.transform(data[self.feature_cate_name])
        else:
            encoded_cols_feature = []
        # Setting up the numpy array
        self.feature_conf_names = encoded_cols_conf+confounder_conti_name+encoded_cols_feature+feature_conti_name
        self.feature_names = encoded_cols_feature+feature_conti_name
        X = data[self.feature_conf_names].to_numpy()
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        self.X = self.scaler.transform(X) # Standardization
        self.z = data[outcome_name].to_numpy() # outcome
        self.p_confound = len(encoded_cols_conf+confounder_conti_name)
        self.p = self.X.shape[1] - self.p_confound
        self.n = self.X.shape[0]
        # Initialize the model
        self.model = linear_slab_spike(p = self.p, n_total = self.n, p_confound = self.p_confound,
                                       init_pi_local_max = 1.0, init_pi_local_min = 0.0,
                                       init_pi_global = 0.5, init_beta_var =1, init_noise_var = 1,
                                       gumbel_softmax_temp = 1, gumbel_softmax_hard = False, 
                                       a1= 0.1, a2=0.1, init_a3= 1.1, init_a4 = 1.1,
                                       b1 = 1.1, b2 = 1.1, init_b3 = 10.0, init_b4 = 0.1, n_E = 1, 
                                       prior_sparsity = True, prior_sparsity_beta = False,
                                       exact_lh = True, device = self.device
                                       ).double().to(self.device)


    def fit(self, lr = 0.5, min_loss_stop_fraction_diff = 0.01, 
            lr_schedule_step = 1000, lr_sch_gamma = 0.8, verbose = True, true_beta = None):
        '''Fit the ZIV model

        Parameters
        ----------
        lr: float, learning rate
        min_loss_stop_fraction_diff: float, the minimum fraction difference between the current losses and the lowest loss to stop the training
        lr_schedule_step: int, the number of interval steps to decay the learning rate
        lr_sch_gamma: float, the decay rate of the learning rate
        verbose: bool, whether to print the loss information for each epochs

        Returns
        -------
        FVE_df: pandas dataframe, result of the total signal profiles: fractional variance explained, global proportion of non-nulls.
        coefficient_df: pandas dataframe, result of the coefficients: posterior mean, posterior standard deviation, posterior probability of being non-null.
        train_prediction: numpy array, the predicted outcome on the training set
        train_error: numpy array, the training error on the training set
        '''
        # data loader for training
        if self.batch_size == None:
            batch_size = self.X.shape[0]
        else:
            batch_size = self.batch_size
        X_train = torch.tensor(self.X)
        z_train = torch.tensor(self.z)
        sim_data = Sim_Dataset(X_train, z_train, device = self.device)
        sim_data_loader = DataLoader(sim_data, batch_size = batch_size)
        if true_beta is None:
            true_beta = np.zeros((self.p,))
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(),lr = lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer = optimizer, gamma = lr_sch_gamma)
        # Training
        self.best_model, train_error, train_prediction, result_dict = train_and_infer(model = self.model, optimizer = optimizer,
                                                                    sim_data_loader = sim_data_loader, 
                                                                    lr_scheduler = lr_scheduler, t = 100, 
                                                                    patience = 200, X = self.X, plot = True, 
                                                                    true_beta = true_beta, lr_schedule_step = lr_schedule_step,
                                                                    verbose = verbose, min_loss_stop_fraction_diff = min_loss_stop_fraction_diff)
        pi_local_est = torch.sigmoid(self.best_model.logit_pi_local)
        beta_est = self.best_model.beta_mu
        # getting the all the features
        beta_est = beta_est.cpu().detach().numpy()
        pi_est = pi_local_est.cpu().detach().numpy()
        feature_names = np.array(self.feature_names)
        dict_result = {'feature_names': feature_names, 
                       'beta':beta_est,'pi': pi_est}
        # Transform to dataframe
        coefficient_df = pd.DataFrame(dict_result)
        FVE_df = {'FVE':result_dict['mean_h_est'], 
                  'FVE_upper':result_dict['h_est_upper'],
                  'FVE_lower':result_dict['h_est_lower'], 
                  'global_non_null':result_dict['global_pi'],
                  'global_non_null_upper':result_dict['global_pi_upper'],
                  'global_non_null_lower':result_dict['global_pi_lower'],
                  'feature_variance':result_dict['mean_var_genetic'],
                  'noise_variance':result_dict['noise_var']
                  }
        return FVE_df, coefficient_df, train_prediction, train_error

    def predict(self, data):
        '''Predict the outcome on the new data
        Parameters
        ----------
        data: dataframe with the same columns as the training dataframe except the outcome column, the new data

        Returns
        -------
        z_pred: numpy array, the predicted outcome
        '''
        # Processs the data into numpy 
        if len(self.confounder_cate_name) > 0:
            encoded_cols_conf = list(self.OHE_conf.get_feature_names_out(self.confounder_cate_name))
            data[encoded_cols_conf] = OHE_conf.transform(data[self.confounder_cate_name])
        else:
            encoded_cols_conf = []
        if len(self.feature_cate_name)>0:
            encoded_cols_feature = list(self.OHE_feature.get_feature_names_out(self.feature_cate_name))
            data[encoded_cols_feature] = OHE_feature.transform(data[self.feature_cate_name])
        else:
            encoded_cols_feature = []
        X = torch.tensor(data[self.feature_conf_names].to_numpy())
        z_pred = self.best_model.predict(X)
        return z_pred