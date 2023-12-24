import sys
sys.path.insert(1,'../src_simulation/')
from sim_helper import sim_helper
sim_data_indictor = False
compare_mcmc = True
if __name__ == '__main__':
    if sim_data_indictor:
        sim_ = sim_helper(n_sim = 10, heritability_l = [0.8, 0.5, 0.25],
                          percent_causal_l = [0.1,0.3], 
                          beta_var_l = [0.1], image_modality = None, random_seed = 1, 
                          path = './', compare_mcmc = compare_mcmc, n_sub_l = [400], 
                          p_sub_l = [50], sim_data = True, rho_l = [0], 
                          linear_outcome_l = [True, False],
                          file_suffix = "sim_X"
                          )
        sim_.full_run()
    else:
        # main simulation using the full dataset from ABCD
        sim_ = sim_helper(n_sim = 10, heritability_l = [0.8, 0.5, 0.25], percent_causal_l = [0.1,0.05, 0.01,0.005], 
                        beta_var_l = [0.1], image_modality = 'tfmri_list', random_seed = 1, path = './', 
                        sim_data = False, compare_mcmc = compare_mcmc,
                        file_suffix = "real_X",
                        data_path = '/Users/juntingren/Desktop/L0_VI_Bayesian_full_experimentation_code/dataset/'
                        )
        sim_.full_run()
