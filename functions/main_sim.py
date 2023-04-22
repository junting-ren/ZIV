import sys
sys.path.insert(1,'../functions/')
from sim_helper import sim_helper


if __name__ == '__main__':
    sim_ = sim_helper( n_sim = 2, heritability_l = [0.5,0.3], percent_causal_l = [0.1], beta_var_l = [1], image_modality = 'tfmri_list', random_seed = 1, path = '../simulation_result')
    sim_.full_run()