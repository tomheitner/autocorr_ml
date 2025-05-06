import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.manifold import TSNE
from tqdm.notebook import tqdm
from datetime import datetime
import pickle

# =========================== AUTOENCODER ====================================

def loss_patch(y_pred, y_true, loss_func):
    if y_pred.shape[-1] > y_true.shape[-1]:
        loss_value = loss_func(y_pred[:, :y_true.shape[-1]], y_true)
    elif y_pred.shape[-1] < y_true.shape[-1]:
        loss_value = loss_func(y_pred, y_true[:, :y_pred.shape[-1]])
    else:
        loss_value = loss_func(y_pred, y_true)
    loss_value /= y_true.shape[-1]
    return loss_value

class ConvolutionalAutoencoderTrainer:
    def __init__(self, autoencoder, device):
        self.network = autoencoder
        self.device = device
        
        #  creating log
        self.log_dict = {
            'training_loss': [],
            'validation_loss': [],
            'training_loss_per_batch': [],
            'validation_loss_per_batch': [],
            'visualizations': []
        } 
        
    def load_checkpoint(self, ckpt_path):
        weights_path = ckpt_path[-1] if isinstance(ckpt_path, list) else ckpt_path
        self.network.load_state_dict(torch.load(weights_path))
        
        if isinstance(ckpt_path, list):
            for path in ckpt_path:
                log_path = path.replace('ckpt', 'log').replace('.pt', '.pkl')
                with open(log_path, 'rb') as f:
                    log_dict = pickle.load(f)
                for key in self.log_dict.keys():                    
                    self.log_dict[key] += log_dict[key]
        else:
            log_path = ckpt_path.replace('ckpt', 'log').replace('.pt', '.pkl')
            with open(log_path, 'rb') as f:
                log_dict = pickle.load(f)
            self.log_dict = log_dict
        
    def train(
        self, 
        lr,
        loss_function, 
        epochs,
        latent_loss_factor,
        train_samples,  # # of train samples to generate each epoch
        val_samples, # # of val samples to generate each epoch
        training_set, 
        validation_set,
        test_set, 
        exp_name='',
        num_to_plot=10,
        verbose=2, 
        log_every=10):
        
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        #  defining weight initialization function
        def init_weights(module):
            if isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                module.bias.data.fill_(0.01)
            elif isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                module.bias.data.fill_(0.01)
        
        #  initializing network weights
        self.network.apply(init_weights)

        #  setting convnet to training mode
        self.network.train()
        self.network.to(self.device)

        if verbose>0: 
            epoch_iter = tqdm(range(epochs))
            # train_iter = tqdm(range(train_samples), leave=False)
            # val_iter = tqdm(range(val_samples), leave=False)
        if verbose == 0:
            epoch_iter = range(epochs)
            # train_iter = range(train_samples)
            # val_iter = range(val_samples)
            
        
        for epoch in epoch_iter:            
            # save checkpoint
            ckpt_save_path = f'./ae_model_checkpoints/{exp_name}_ae_model_{str(datetime.now().date())}_ckpt_{int(epoch*train_samples)}.pt'
            torch.save(self.network.state_dict(), ckpt_save_path)
            
            # save checkpoint's logs
            log_ckpt_save_path = f'./ae_model_checkpoints/{exp_name}_ae_model_{str(datetime.now().date())}_log_{int(epoch*train_samples)}.pkl'
            with open(log_ckpt_save_path, 'wb') as f:
                pickle.dump(self.log_dict, f)
            
            if verbose>0: 
                # epoch_iter.set_description(f'Train Loss: {loss.item():.4f} Val Loss: {_neg.item():.4f}')
                epoch_iter.update()
                
            if verbose>0: 
                train_iter = tqdm(range(train_samples), leave=False)
                val_iter = tqdm(range(val_samples), leave=False)
            if verbose == 0:
                train_iter = range(train_samples)
                val_iter = range(val_samples)
                
            #------------
            #  TRAINING
            #------------
            
            for sample_idx in train_iter:
                
                peak1, peak2, _ = next(training_set)
                
                #  zeroing gradients
                self.optimizer.zero_grad()
                #  sending peak to self.device
                peak1 = peak1.float().to(self.device)
                peak2 = peak2.float().to(self.device)
                
                #  reconstructing peak
                latent1 = self.network.encoder(peak1)
                latent2 = self.network.encoder(peak2)
                
                peak1_hat = self.network.decoder(latent1)
                peak2_hat = self.network.decoder(latent2)
                
                #  computing loss
                # print(f"out shape {output.shape}, in shape {peak.shape}")
                latent_loss1 = loss_patch(latent1, latent2, loss_function)
                recon_loss1 = loss_patch(peak1_hat, peak1, loss_function)
                recon_loss2 = loss_patch(peak2_hat, peak2, loss_function)
                loss = recon_loss1 + recon_loss2 + latent_loss_factor*latent_loss1
                #  calculating gradients
                loss.backward()
                #  optimizing weights
                self.optimizer.step()

                #--------------
                # LOGGING
                #--------------
                loss_value = loss.item()
                self.log_dict['training_loss_per_batch'].append(loss_value)
                
                if verbose>0 and sample_idx%log_every == 0: 
                    train_iter.set_description(f'Train Loss: {loss_value:.4f}')
                    train_iter.update()

            #--------------
            # VALIDATION
            #--------------
            for sample_idx in val_iter:    
                peak1, peak2, _ = next(validation_set)
                with torch.no_grad():                
                    #  zeroing gradients
                    self.optimizer.zero_grad()
                    #  sending peak to self.device
                    peak1 = peak1.float().to(self.device)
                    peak2 = peak2.float().to(self.device)

                    #  reconstructing peak
                    latent1 = self.network.encoder(peak1)
                    latent2 = self.network.encoder(peak2)

                    peak1_hat = self.network.decoder(latent1)
                    peak2_hat = self.network.decoder(latent2)

                    #  computing loss
                    # print(f"out shape {output.shape}, in shape {peak.shape}")
                    latent_loss1 = loss_patch(latent1, latent2, loss_function)
                    recon_loss1 = loss_patch(peak1_hat, peak1, loss_function)
                    recon_loss2 = loss_patch(peak2_hat, peak2, loss_function)
                    val_loss = recon_loss1 + recon_loss2 + latent_loss_factor*latent_loss1

                #--------------
                # LOGGING
                #--------------
                val_loss_value = val_loss.item()
                self.log_dict['validation_loss_per_batch'].append(val_loss_value)
                if verbose>0 and sample_idx%log_every == 0: 
                    val_iter.set_description(f'Val Loss: {val_loss_value:.4f}')
                    val_iter.update()               
                
            epoch_average_train_loss = np.mean(self.log_dict['training_loss_per_batch'])
            epoch_average_val_loss = np.mean(self.log_dict['validation_loss_per_batch'])
            if verbose>0:
                train_iter.set_description(f'Epoch Average - Train Loss: {epoch_average_train_loss:.4f}')
                train_iter.update()
                val_iter.set_description(f'Epoch Average - Val Loss: {epoch_average_val_loss:.4f}')
                val_iter.update()
            
            self.log_dict['training_loss'].append(epoch_average_train_loss)
            self.log_dict['validation_loss'].append(epoch_average_val_loss)
            
            # reset
            self.log_dict['training_loss_per_batch'] = []
            self.log_dict['validation_loss_per_batch'] = []
            #--------------
            # VISUALISATION
            #--------------
            if verbose > 1:
                self.plot_loss()
                
            image_array = self.reconstruction_plots(test_set, verbose, num_to_plot=num_to_plot)
            self.log_dict['visualizations'].append(image_array)
    def autoencode(self, x):
        return self.network(x)

    def encode(self, x):
        encoder = self.network.encoder
        return encoder(x)
  
    def decode(self, x):
        decoder = self.network.decoder
        return decoder(x)
    
    def plot_loss(self):
        plt.figure(figsize=(7, 3))
        plt.plot(self.log_dict['training_loss'], label="Train Loss")
        plt.plot(self.log_dict['validation_loss'], label="Val Loss")
        plt.legend()
        plt.grid()
        plt.show()
        
    def reconstruction_plots(self, dataset, verbose, num_to_plot=10):
        fig, ax = plt.subplots(num_to_plot, 2, figsize=(15, 1.5*num_to_plot))
        for i in range(num_to_plot):
            peak, _, _ = next(dataset)
            #  sending test peak to self.device
            peak = peak.float().to(self.device)
            with torch.no_grad():
                #  reconstructing test peak
                reconstructed_peak = self.network(peak)
                #  sending reconstructed and peak to cpu to allow for visualization
                reconstructed_peak = reconstructed_peak.cpu()
                peak = peak.cpu()

            #  visualisation
            signal_to_plot = abs(peak[0] + 1j*peak[1])
            ax[i][0].plot(signal_to_plot)
            signal_hat_to_plot = abs(reconstructed_peak[0] + 1j*reconstructed_peak[1])
            ax[i][1].plot(signal_hat_to_plot)

        canvas = FigureCanvas(fig)
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image_array = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)
        plt.suptitle('Original/Reconstructed')
        
        if verbose > 2:
            plt.show()
        else:
            plt.close()
        return image_array
    
    def reconstruction_plots_history(self):
        epochs = len(self.log_dict['visualizations'])
        for epoch_idx in range(epochs):
            plt.figure(figsize=(20, 15))
            plt.imshow(self.log_dict['visualizations'][epoch_idx])
            plt.title(f"Epoch {epoch_idx}")
            plt.show()

    def tsne_plots(self, test_set, device, num_samples=100):    
        for name, habub in list(self.network.encoder.net.named_children())[::-1]:
            if isinstance(habub, nn.Conv1d):
                num_latent_channels = habub.out_channels
        # passing through model to get encoded results
        self.network.eval()
        res_list = []
        trg_list = []
        for sample_idx in tqdm(range(num_samples)):
            # ----- forward pass -----
            peak, _, trg = next(test_set)
            # peak = peak.to(torch.device('cpu'))
            peak = peak.to(device).float()
            trg_list.append(trg)
            res = self.encode(peak.unsqueeze(0)).detach().cpu().squeeze().numpy()
            res_list.append(res)
            # ----- forward pass -----

        # padding for TSNE
        max_len = max([len(res_list[sample_idx][0]) for sample_idx in range(num_samples)])
        res_list_padded = np.zeros((num_samples, num_latent_channels, max_len))
        for channel_idx in range(num_latent_channels):
            for sample_idx in range(num_samples):
                current_len = len(res_list[sample_idx][channel_idx])
                if (max_len-current_len) % 2 == 0:
                    res_list_padded[sample_idx][channel_idx] = np.pad(res_list[sample_idx][channel_idx], pad_width=((max_len-current_len)//2, (max_len-current_len)//2))
                else:
                    res_list_padded[sample_idx][channel_idx] = np.pad(res_list[sample_idx][channel_idx], pad_width=((max_len-current_len)//2, (max_len-current_len)//2 + 1))

        # TSNE plots
        fig, ax = plt.subplots(num_latent_channels//4, 4, figsize=(25, num_latent_channels))

        for channel_idx in range(num_latent_channels):
            tsne = TSNE(n_components=2, init='pca')
            tsne_result = tsne.fit_transform(res_list_padded[:, channel_idx, :])


            for signal_type in np.unique(trg_list):
                tsne_result_signal_type = tsne_result[trg_list == signal_type]    
                ax[channel_idx//4][channel_idx%4].scatter(
                    tsne_result_signal_type[:, 0],
                    tsne_result_signal_type[:, 1],
                    label=test_set.gen_objs[signal_type]
                )
            ax[channel_idx//4][channel_idx%4].set_title(f"channel {channel_idx}")
        plt.legend()
        plt.show()
# =========================== AUTOENCODER ====================================



