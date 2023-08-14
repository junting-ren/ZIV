import sys
sys.path.insert(1,'../src_simulation/')
from sim_helper import sim_helper

if __name__ == '__main__':
    # main simulation using the full dataset
    sim_ = sim_helper( n_sim = 500, heritability_l = [0.8, 0.5, 0.25,0.05], percent_causal_l = [0.1,0.05, 0.01,0.005], beta_var_l = [0.1], image_modality = 'tfmri_list', random_seed = 1, path = './')
    sim_.full_run()
    # supporting simulation using sub dataset for comparison between VI and MCMC
    sim_ = sim_helper( n_sim = 500, heritability_l = [0.5], percent_causal_l = [0.1], beta_var_l = [0.1], image_modality = 'tfmri_list', random_seed = 1, path = './', compare_mcmc = True, n_sub_l = [1000], p_sub_l = [50,100,200,500])
    sim_.full_run()