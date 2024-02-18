import sys
sys.path.insert(1,'./')
sys.path.insert(1,'../src_VI/')
sys.path.insert(1,'../src_MCMC/')
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import data_sim
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV,RidgeCV
import pyreadr
from sklearn.model_selection import ParameterGrid
import os
import numpy as np
import time
from gen_GCTA_GRM import gen_GCTA_GRM
import subprocess
from scipy.linalg import pinv
from scipy.stats import pearsonr

# 1. Simulate the X data with fix p and max n
# 2. Write the GMR matrix
# 3. Function to do one simluation:
#     a. takes in n and beta variance, percent_causal, linear_outcome index
#     b. if n< n_max, randomly select n and save the index
#     c. Generate y using the sub X
#     d. Run GCTA with inputs specifying the index for the y and the fixed GRM

def one_run(train_X, test_X, h, percent_causal, beta_var,rho, linear_outcome = False, n_sub = None, p_sub = None,
            gcta_path = "./gcta", grm_path = "./grm", tmp_loc = "./"
            ):
    #import pdb; pdb.set_trace()
    n = train_X.shape[0]
    n_sub_train = int(n_sub*0.8)
    n_sub_test = n_sub - n_sub_train
    train_index = np.random.choice(range(0, n), size = n_sub_train, replace = False)
    X_train = train_X[train_index,:]
    X_test = test_X[:n_sub_test]
    p_confound = 0
    p = X_train.shape[1]-p_confound
    p_causal = int(p*percent_causal)
    if p_sub is not None and p_sub<p:
        p_index = np.random.choice(range(0, p), size = p_sub, replace = False)
        X_train = X_train[:, p_index]
        X_test = X_test[:, p_index]
    X = np.row_stack([X_train, X_test])
    # simulate data
    sim_class = data_sim.sim_tobit_data(n = n, p = p, p_causal = p_causal, p_confound =0, rho = None, var = None,
                                        n_matrix = 1,h = h, bias = 0, Xs = [X], scale_lambda =None, beta_var= beta_var)
    z, X, _, latent_mean, var_genetic, var_total, true_beta, y_star = sim_class.gen_data(seed = None)
    ##################################
    ### if we need linear outcome  
    ##################################
    if linear_outcome:
        z = y_star
    ##################################
    X_train = X[:n_sub_train,:]
    X_test = X[n_sub_train:, :]
    z_train = z[:n_sub_train]
    z_test = z[n_sub_train:]
    ##############################################
    ### Run GCTA
    ##############################################
    # Generate BRM
    BRM = np.corrcoef(X_train)
    # Also need to get the inverse BRM first for the BLUP
    WtAinv = np.transpose(X_train) @ pinv(BRM)  / X_train.shape[0]
    result_dict = {}
    ##############################################
    # All observed z
    ###############################################
    random_file_name = str(np.random.uniform(size = 1))
    pheno_file = tmp_loc+random_file_name+"train.brm.pheno"
    # Write phenotype data to file
    with open(pheno_file, 'w') as fid_pheno:
        for idx, val in enumerate(z_train):
            fid_pheno.write(f'{(train_index+1)[idx]}\t{(train_index+1)[idx]}\t{val}\n')
    cmd = f"{gcta_path} --thread-num 2 --grm-gz {grm_path} --pheno {pheno_file} --reml --reml-pred-rand --out {pheno_file}tmp"
    subprocess.run(cmd, check=True, shell=True)
    with open(f"{pheno_file}tmp.hsq", 'r') as fid:
        # Skipping headers and reading data
        headers = next(fid).split()
        sigma_a, sigma_e, _, h2 = [float(line.split()[1]) for line in fid if line.split()[0] in ['V(G)', 'V(e)', 'Vp', 'V(G)/Vp']]
    # Read predicted genetic values
    g = pd.read_csv(f"{pheno_file}tmp.indi.blp", delim_whitespace=True, header=None)
    u = WtAinv @ g.iloc[:, 3]  # Assuming the genetic values are in the fourth column
    z_pred = X_test @ u
    mae = np.mean(np.abs(z_pred*(z_pred>0) - z_test))
    os.remove(pheno_file)
    os.remove(f"{pheno_file}tmp.log")
    os.remove(f"{pheno_file}tmp.indi.blp")
    os.remove(f"{pheno_file}tmp.hsq")
    ##############################################
    # truncated z
    ###############################################
    if not linear_outcome:
        random_file_name = str(np.random.uniform(size = 1))
        pheno_file = tmp_loc+random_file_name+"train.brm.pheno"
        # Write phenotype data to file
        with open(pheno_file, 'w') as fid_pheno:
            for idx, val in enumerate(z_train):
                if val>0:
                    fid_pheno.write(f'{(train_index+1)[idx]}\t{(train_index+1)[idx]}\t{val}\n')
        cmd = f"{gcta_path} --thread-num 2 --grm-gz {grm_path} --pheno {pheno_file} --reml --reml-pred-rand --out {pheno_file}tmp"
        subprocess.run(cmd, check=True, shell=True)
        with open(f"{pheno_file}tmp.hsq", 'r') as fid:
            # Skipping headers and reading data
            headers = next(fid).split()
            sigma_a, sigma_e, _, h2_non_truncated = [float(line.split()[1]) for line in fid if line.split()[0] in ['V(G)', 'V(e)', 'Vp', 'V(G)/Vp']]
        # Read predicted genetic values
        g = pd.read_csv(f"{pheno_file}tmp.indi.blp", delim_whitespace=True, header=None)
        BRM = np.corrcoef(X_train[z_train>0])
        # Also need to get the inverse BRM first for the BLUP
        WtAinv = np.transpose(X_train[z_train>0]) @ pinv(BRM)  / X_train[z_train>0].shape[0]
        u = WtAinv @ g.iloc[:, 3]  # Assuming the genetic values are in the fourth column
        z_pred = X_test @ u
        mae_non_truncated = np.mean(np.abs(z_pred*(z_pred>0) - z_test))
        os.remove(pheno_file)
        os.remove(f"{pheno_file}tmp.log")
        os.remove(f"{pheno_file}tmp.indi.blp")
        os.remove(f"{pheno_file}tmp.hsq")
        result_dict['FVE_gcta_non_truncated'] = h2_non_truncated
        result_dict['mae_gcta_non_truncated'] = mae_non_truncated
    else:
        result_dict['FVE_gcta_non_truncated'] = None
        result_dict['mae_gcta_non_truncated'] = None
    #import pdb; pdb.set_trace()
    result_dict['n'] = n_sub_train
    result_dict['p'] = p_sub
    result_dict['linear_outcome'] = linear_outcome
    result_dict['true_h'] = h
    result_dict['true_pi'] = percent_causal
    result_dict['beta_var'] = beta_var
    result_dict['rho'] = rho
    result_dict['FVE_gcta'] = h2
    result_dict['mae_gcta'] = mae
    return [z_train,z_test, pd.DataFrame(result_dict,index=[0])]

def one_run_wrapper(kwargs):
    try: 
        result = one_run(**kwargs)
    except Exception as e:
        print(e)
        result = [np.array([]), np.array([]), pd.DataFrame()]
    return result


class sim_helper_GCTA(object):
    def __init__(self, n_sim, heritability_l, percent_causal_l, beta_var_l, image_modality, linear_outcome_l = [False],
                 random_seed = 1, path = '', file_suffix = '', n_sub_l = None, p_sub_l = None, 
                 sim_data = False, rho_l = [None], data_path = './', gcta_path = "./"):
        self.n_sim = n_sim
        self.heritability_l = heritability_l
        self.percent_causal_l = percent_causal_l
        self.beta_var_l = beta_var_l
        self.image_modality = image_modality
        self.random_seed = random_seed
        self.path = path
        self.n_sub_l = n_sub_l
        self.p_sub_l = p_sub_l
        self.sim_data = sim_data
        self.rho_l = rho_l
        self.linear_outcome_l = linear_outcome_l
        self.file_suffix = file_suffix
        self.data_path = data_path
        self.gcta_path = gcta_path
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
            if os.path.exists(self.data_path+'max_n_sim_X_train.csv'):
                data_save_train = pd.read_csv(self.data_path+'max_n_sim_X_train.csv')
                data_save_test = pd.read_csv(self.data_path+'max_n_sim_X_test.csv')
            else:
                sim_class = data_sim.sim_tobit_data(n = max(self.n_sub_l), p = max(self.p_sub_l), p_causal = 1, p_confound =0, rho = 0, var = 1,
                                            n_matrix = 1,h = 0.5, bias = 0, Xs = None, scale_lambda =None, beta_var= 0.1)
                _, X, _, _, _, _, _, _ = sim_class.gen_data(seed = None)
                n = X.shape[0]
                indices = np.random.permutation(n)
                split = int(n*0.8)
                train_index, test_index = indices[:split], indices[split:]
                # Save the feature matrix
                data_save_train = pd.DataFrame(X[train_index,:])
                data_save_train.to_csv(self.data_path+ 'max_n_sim_X_train.csv', index = False)
                data_save_test = pd.DataFrame(X[test_index,:])
                data_save_test.to_csv(self.data_path+'max_n_sim_X_test.csv', index = False)
        else:
            self.load_clean_data()
            n = self.data.shape[0]
            indices = np.random.permutation(n)
            split = int(n*0.8)
            train_index, test_index = indices[:split], indices[split:]
            # Save the feature matrix
            data_save_train = pd.DataFrame(self.data[train_index,:])
            data_save_train.to_csv(os.path.join(self.data_path, self.image_modality+'_train_standardized_features.csv'), index = False)
            data_save_test = pd.DataFrame(self.data[test_index,:])
            data_save_test.to_csv(os.path.join(self.data_path, self.image_modality+'_test_standardized_features.csv'), index = False)
        ## Generate the GRM information
        if not os.path.exists(f'{self.data_path}saved_grm.grm.gz'):
            grm = np.corrcoef(data_save_train)
            M = data_save_train.shape[1]
            grm_info = gen_GCTA_GRM(M, grm, outpath = self.data_path)
        #import pdb; pdb.set_trace()
        # setting up the parameter grid
        data_save_train = np.array(data_save_train)
        data_save_test = np.array(data_save_test)
        if self.n_sub_l is None:
            self.n_sub_l = [n]
        if self.p_sub_l is None:
            self.p_sub_l = [self.data.shape[1]]
        param_grid = {'train_X': [data_save_train], 'test_X': [data_save_test],'h':self.heritability_l,
                      'percent_causal': self.percent_causal_l, 'beta_var': self.beta_var_l, 
                      'n_sub':self.n_sub_l,'p_sub':self.p_sub_l, 'rho':self.rho_l , 'linear_outcome':self.linear_outcome_l,
                      'gcta_path': [self.gcta_path], 'grm_path': [f'{self.data_path}saved_grm'], 'tmp_loc':[self.data_path]
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
    
if __name__ == "__main__":
        sim_ = sim_helper_GCTA(n_sim = 200, heritability_l = [0.8,0.5,0.25],
                        percent_causal_l = [0.1,0.3], 
                        beta_var_l = [0.1], image_modality = None, random_seed = 1, 
                        path = '/mnt/c/Users/juntingr/Desktop/L0_VI_Bayesian_approx/TCGA_result/', 
                        n_sub_l = [200,400,1000,2000,4000,8000], 
                        p_sub_l = [400], sim_data = True, rho_l = [0], 
                        linear_outcome_l = [False, True],
                        file_suffix = "TCGA", 
                        data_path = '/mnt/c/Users/juntingr/Desktop/L0_VI_Bayesian_approx/dataset/saved_sim_X/',
                        gcta_path = "/mnt/c/Users/juntingr/Desktop/gcta-1.94.1-linux-kernel-3-x86_64/gcta-1.94.1")
        sim_.full_run()