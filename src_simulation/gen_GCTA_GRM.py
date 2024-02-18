import numpy as np
import gzip
import pandas as pd
import os 
def gen_GCTA_GRM(M, GRM, outpath):
    """
    Generate the lower triangle indexed GRM text files for GCTA or OSCA reml analysis.
    
    Parameters:
    - info: A dictionary to store relevant information.
    - M: Number of SNPs/Vertices used for calculating the GRM.
    - GRM: Relatedness matrix.
    - outpath: Output path, including prefix.
    
    Returns:
    - info: Updated dictionary with additional path information.
    """
    nsubj = GRM.shape[0]
    info = {}
    info['grm_path'] = f'{outpath}saved_grm.grm.gz'
    info['grm_id_path'] = f'{outpath}saved_grm.grm.id'
    rowID = range(1, nsubj+1)
    # Print info
    print(info)
    
    tcells = nsubj * (nsubj + 1) // 2
    tid = 1
    
    # Writing .grm file
    with open(f'{outpath}.grm', 'w') as fid:
        for idx in range(1, nsubj + 1):
            for idy in range(1, idx + 1):
                print(f'Writing {tid} of {tcells} cells', end='\r')
                fid.write(f'{idx}\t{idy}\t{M}\t{GRM[idx-1, idy-1]}\n')
                tid += 1
    
    print('\nGzipping the grm')
    with open(f'{outpath}.grm', 'rb') as f_in, gzip.open(info['grm_path'], 'wb') as f_out:
        f_out.writelines(f_in)
    #os.remove(f'{outpath}.grm')
    
    # Writing .grm.id file
    print('Writing the ID file')
    with open(info['grm_id_path'], 'w') as fid:
        for idx in range(nsubj):
            fid.write(f'{rowID[idx]}\t{rowID[idx]}\n')
    
    print('Done!')
    return info

if __name__ == "__main__":
    X = pd.read_csv("./saved_sim_X/max_n_sim_X_train.csv")
    grm = np.corrcoef(X)
    M = X.shape[1]
    info = gen_GCTA_GRM(M, grm, outpath= "./saved_sim_X/")
    print(info)
