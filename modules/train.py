from torch import optim
import numpy as np

from matplotlib import pyplot as plt

from tqdm.notebook import tqdm
import torch

from datetime import datetime
import pickle


torch.autograd.set_detect_anomaly(True)


def normalize(x):
    x_norm = (x - x.min()) / (x.max() - x.min())
    
    return x_norm

def neg_cos_sim(p, z):  # negative cosine similarity   
    z.detach()  # stop gradient
    
    p = normalize(p)
    z = normalize(z)
    return -(p*z).mean(dim=-1)

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
    losses=[],
    losses_neg=[],
    verbose=0,
    log_every=5,    # save loss value (in batches)
    save_every=1000, # model checkpoint (in epochs)
    log_to_file=False,  # save losses to file to see progress while training
):
    
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
        except Exception as e:
            print(e)
            print(peak1.shape)
            print(peak2.shape)
            print(label)
        # ----- forward pass -----
        
        # ----- update weights -----
        loss = pos_weight*neg_cos_sim(res1, res2) + 1 / (res1**2 + res2**2).mean()
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
        except Exception as e:
            print(e)
            print(peak1.shape, label1)
            print(peak2.shape, label2)
        # ----- forward pass -----

        # ----- update weights -----
        loss_neg = -neg_weight*neg_cos_sim(res1, res2) + 1 / (res1**2 + res2**2).mean()
        loss_neg.backward()
        optimizer.step()
        # ----- update weights -----
        
        


        if losses is not None and sample_idx%log_every == 0: 
            losses.append(loss.item())
            if log_to_file:
                with open('./progress_log/pos_loss.pkl', 'wb') as f:
                    pickle.dump(losses, f)
        if losses_neg is not None and sample_idx%log_every == 0: 
            losses_neg.append(loss_neg.item())
            
            if log_to_file:
                with open('./progress_log/neg_loss.pkl', 'wb') as f:
                    pickle.dump(losses_neg, f)

        # ----- verbose stuff -----
        if verbose>0: 
            if sample_idx%log_every == 0: samples_iter.set_description(f'PosLoss: {loss.item():.4f} NegLoss: {loss_neg.item():.4f}')
            samples_iter.update()

        if verbose>1 and sample_idx%plot_every == 0:
            # fig.clear()

            # plt.plot(losses)
            plt.semilogy(np.arange(len(losses))*log_every, losses)
            plt.xlabel('#Samples')
            plt.ylabel('Loss')
            plt.show()
            # fig.canvas.draw()
            # fig.canvas.flush_events()
        # time.sleep(0.1)
        # ----- verbose stuff -----

        if sample_idx == 0:
            architecture_save_path = f'./model_checkpoints/model_{str(datetime.now().date())}.txt'
            with open(architecture_save_path, 'w') as f:
                f.write(str(model))
                
        if sample_idx%save_every == 0:
            ckpt_save_path = f'./model_checkpoints/model_{str(datetime.now().date())}_ckpt_{sample_idx}.pt'
            torch.save(model.state_dict, ckpt_save_path)