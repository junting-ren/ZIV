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
    
def train_and_infer(model, optimizer, sim_data_loader, lr_scheduler, t, patience, 
                    X, plot = False, true_beta = None, verbose = False,lr_schedule_step = 1000,
                    min_loss_stop_fraction_diff = 0.01
                   ):
    p_cur = 0 
    min_avg_loss = float('inf')
    losses = []
    best_model = None
    for i in range(20000):
        loss_epoch = 0
        for j, (X_batch, y_batch) in enumerate(sim_data_loader):
            #import pdb; pdb.set_trace()
            loss = -model.ELBO(X_batch,y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            if np.isnan(loss.item()):
                return model,  {'mean_h_est': [-1], 'h_est_upper': [-1], 'h_est_lower': [-1], 
                'mean_var_genetic': [-1], 'noise_var': [-1], 
                'global_pi':[-1], 'global_pi_upper':[-1], 'global_pi_lower':[-1]}
        losses.append(loss_epoch)
        if i % lr_schedule_step== 0:
            lr_scheduler.step()
        if i % 500 == 0:
            if verbose:
                print(f'At iteration {i}, the loss is {loss.item()}')
        if i > t:
            cur_avg_loss = np.mean(losses[-t:-1])
            if cur_avg_loss < min_avg_loss and abs(min_avg_loss-cur_avg_loss)>cur_avg_loss*min_loss_stop_fraction_diff:
                min_avg_loss = cur_avg_loss
                p_cur = 0
                best_model = copy.deepcopy(model)
            else:
                p_cur += 1
        if p_cur > patience:
            break
    if best_model is None:
        best_model = model
        return best_model, best_model.inference(est_mean= est_mean, num_samples = num_samples, plot = plot, true_beta = true_beta)
    
    num_samples = 4000
    sample_beta = best_model.sample_beta(num_samples)
    est_mean_l = []
    error_l = []
    point_est_l = []
    for j, (X_batch, y_batch) in enumerate(sim_data_loader):
        batch_mean, batch_error, batch_point = best_model.cal_mean_batch(X_batch, sample_beta, y_batch)
        est_mean_l.append(batch_mean)
        error_l.append(batch_error)
        point_est_l.append(batch_point)
    #import pdb; pdb.set_trace()
    est_mean = np.concatenate(est_mean_l)
    error = np.concatenate(error_l)
    point_est = np.concatenate(point_est_l)
    return best_model, error, point_est, best_model.inference(est_mean= est_mean, num_samples = num_samples, plot = plot, true_beta = true_beta)