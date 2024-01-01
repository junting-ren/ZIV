import sys
sys.path.insert(1,'../src_simulation/')
from sim_helper import sim_helper
compare_mcmc = True
if __name__ == '__main__':
    # sim_ = sim_helper(n_sim = 200, heritability_l = [0.8, 0.5, 0.25],
    #                       percent_causal_l = [0.1,0.3], 
    #                       beta_var_l = [0.1], image_modality = None, random_seed = 1, 
    #                       path = './', compare_mcmc = compare_mcmc, n_sub_l = [200,400,1000,2000,4000], 
    #                       p_sub_l = [400], sim_data = True, rho_l = [0], 
    #                       linear_outcome_l = [True, False],
    #                       file_suffix = "sim_X"
    #                       )
    # sim_.full_run()
    # main simulation using the full dataset from ABCD
    sim_real_X = sim_helper(n_sim = 200, heritability_l = [0.8, 0.5, 0.25], percent_causal_l = [0.1,0.05, 0.01,0.005], 
                        beta_var_l = [0.1], image_modality = 'tfmri_list', random_seed = 1, path = './', 
                        sim_data = False, compare_mcmc = compare_mcmc,
                        file_suffix = "real_X",
                        linear_outcome_l = [True, False],
                        data_path = '../dataset/'
                        )
    sim_real_X.full_run()
