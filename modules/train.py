from torch import optim
import torch.nn.functional as F
import numpy as np


from matplotlib import pyplot as plt

# from tqdm.notebook import tqdm
from tqdm import tqdm
import torch

from datetime import datetime
import pickle


import mlflow





torch.autograd.set_detect_anomaly(True)


# def neg_cos_sim(p, z):  # negative cosine similarity   
#     z.detach()  # stop gradient
    
#     p = normalize(p)
#     z = normalize(z)
#     return -(p*z).sum(dim=-1)#.mean()


def fit(
    model,
    train_ds,
    pos_weight=1,
    neg_weight=1,
    lr=0.0001,
    num_samples=100,
    device=torch.device('cpu'),
    verbose=0,
    log_every=5,    # save loss value (in batches)
    save_every=1000, # model checkpoint (in epochs)
    run_name=None,
    mlflow_url="http://localhost:5000",
):
    mlflow.set_tracking_uri(mlflow_url)
    run_name = f"model_{str(datetime.now().date())}" if run_name is None else run_name
    mlflow.set_experiment("XcorrML")

    with mlflow.start_run(run_name=run_name):
        model.train()
        model.to_device(device)
        # optimizer = optim.SGD(model.parameters(), lr=lr)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        if verbose>0: 
            samples_iter = tqdm(range(num_samples))
        if verbose == 0:
            samples_iter = range(num_samples)
            
        
        # TRAIN LOOP
        for sample_idx in samples_iter:
            
            '''
            POSITIVE EXAMPLES
            '''
            optimizer.zero_grad()       
            # ----- forward pass -----
            peak1, peak2, label = next(train_ds)
            peak1.to(device)
            peak2.to(device)
            try:
                res1 = model(peak1.unsqueeze(0))
                res2 = model(peak2.unsqueeze(0))
                res2.detach()  # stop gradient
            except Exception as e:
                print(e)
                print(peak1.shape)
                print(peak2.shape)
                print(label)
            # ----- forward pass -----
            
            # ----- update weights -----
            regularization_factor = 1 / (res1**2).mean() + 1 / (res2**2).mean()  # to make sure it doesn't collapse to zero
            # the negative at the start because we want the value to be low when res1, res2 are similar
            loss = -pos_weight*F.cosine_similarity(res1, res2) + regularization_factor
            loss.backward()
            optimizer.step()
            # ----- update weights -----
            
            '''
            NEGATIVE EXAMPLES
            '''
            optimizer.zero_grad()
            
            # ----- forward pass -----
            peak1, _, label1 = next(train_ds)
            
            # we want different type of signal
            label2 = label1
            while label1 == label2:
                peak2, _, label2 = next(train_ds)
            
            peak1 = peak1.to(device)
            peak2 = peak2.to(device)
            try:
                res1 = model(peak1.unsqueeze(0))
                res2 = model(peak2.unsqueeze(0))
                res2.detach()  # stop gradient
            except Exception as e:
                print(e)
                print(peak1.shape, label1)
                print(peak2.shape, label2)
            # ----- forward pass -----

            # ----- update weights -----
            regularization_factor = 1 / (res1**2).mean() + 1 / (res2**2).mean()  # to make sure it doesn't collapse to zero
            loss_neg = neg_weight*F.cosine_similarity(res1, res2) + regularization_factor
            loss_neg.backward()
            optimizer.step()
            # ----- update weights -----
            
            

            if sample_idx%log_every == 0: 
                mlflow.log_metric("Loss Positive Examples", loss.item(), step=sample_idx)
                mlflow.log_metric("Loss Negative Examples", loss_neg.item(), step=sample_idx)

                # ----- verbose stuff -----
                if verbose>0: 
                    samples_iter.set_description(f'PosLoss: {loss.item():.4f} NegLoss: {loss_neg.item():.4f}')
                    samples_iter.update()

            if sample_idx == 0:
                architecture_save_path = f'./model_checkpoints/model_{str(datetime.now().date())}.txt'
                with open(architecture_save_path, 'w') as f:
                    f.write(str(model))
                    
            if sample_idx%save_every == 0:
                ckpt_save_path = f'./model_checkpoints/model_{str(datetime.now().date())}_ckpt_{sample_idx}.pt'
                torch.save(model.state_dict, ckpt_save_path)