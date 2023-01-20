import copy
from torch.utils.data import Dataset
import numpy as np
class Sim_Dataset(Dataset):
    
    def __init__(self, X, y, device = 'cpu'):
        self.X = X
        self.y = y
        self.device = device
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx,:].double().to(self.device), self.y[idx].double().to(self.device)
    
def train_and_infer(model, optimizer, sim_data_loader, lr_scheduler, t, patience, X, plot = False, true_beta = None, verbose = False):
    p_cur = 0 
    min_avg_loss = float('inf')
    losses = []
    B = len(sim_data_loader)
    best_model = None
    for i in range(20000):
        loss_epoch = 0
        for j, (X_batch, y_batch) in enumerate(sim_data_loader):
            #import pdb; pdb.set_trace()
            loss = -model.ELBO(X_batch,y_batch, B)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            if np.isnan(loss.item()):
                return model,  {'mean_h_est': [-1], 'h_est_upper': [-1], 'h_est_lower': [-1], 
                'mean_var_genetic': [-1], 'noise_var': [-1], 
                'global_pi':[-1], 'global_pi_upper':[-1], 'global_pi_lower':[-1]}
        losses.append(loss_epoch)
        if i % 1000 == 0:
            lr_scheduler.step()
            if verbose:
                print(f'At iteration {i}, the loss is {loss.item()}')
        if i > t:
            cur_avg_loss = np.mean(losses[-t:-1])
            if cur_avg_loss < min_avg_loss:
                min_avg_loss = cur_avg_loss
                p_cur = 0
                best_model = copy.deepcopy(model)
            else:
                p_cur += 1
        if p_cur > patience:
            break
    if best_model is None:
        best_model = model
        ret
    return best_model, best_model.inference(X = X.double(),  num_samples = 1000, plot = plot, true_beta = true_beta)