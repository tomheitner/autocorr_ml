# ========================== IMPORTS ==================================
import torch
from torch import nn
from collections import OrderedDict
import numpy as np
# ========================== IMPORTS ==================================

# =========================== AUTOENCODER ====================================
#  defining encoder
class Encoder(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels=16,
        act_fn=nn.ReLU(),
        kernel_size=5,
        stride=1,
        padding=1,
        dilation=1,
        # maxpool_kernel_size=2,
        # drouput_p=0.2
    ):
        super().__init__()
        
        self.iNorm = nn.InstanceNorm1d(num_features=in_channels)
        adaptive_pooling = nn.AdaptiveAvgPool1d(output_size=1)

        
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation), # (32, 32)
            act_fn,
            nn.Conv1d(out_channels, 2*out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation), # (16, 16)
            act_fn,
            nn.Conv1d(2*out_channels, 2*out_channels, kernel_size, padding=padding, dilation=dilation),
            act_fn,
            nn.Conv1d(2*out_channels, 4*out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation), # (8, 8)
            act_fn,
            nn.Conv1d(4*out_channels, 4*out_channels, kernel_size, padding=padding, dilation=dilation),
            act_fn,
            # nn.AdaptiveAvgPool1d(output_size=1),
            # nn.Flatten(),
        )

    def forward(self, x):
        # x = x.view(-1, 3, 32, 32)
        x = self.iNorm(x)
        output = self.net(x)
        return output

#  defining decoder
class Decoder(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels=16,
        act_fn=nn.ReLU(),
        kernel_size=5,
        stride=1,
        padding=1,
        dilation=1,
        output_padding=0
        # maxpool_kernel_size=2,
        # drouput_p=0.2
    ):
        super().__init__()

        self.out_channels = out_channels


        self.conv = nn.Sequential(
            nn.ConvTranspose1d(4*out_channels, 4*out_channels, kernel_size, padding=padding, dilation=dilation), # (8, 8)
            act_fn,
            nn.ConvTranspose1d(4*out_channels, 2*out_channels, kernel_size, padding=padding, dilation=dilation,
                               stride=stride, output_padding=output_padding), # (16, 16)
            act_fn,
            nn.ConvTranspose1d(2*out_channels, 2*out_channels, kernel_size, padding=padding, dilation=dilation),
            act_fn,
            nn.ConvTranspose1d(2*out_channels, out_channels, kernel_size, padding=padding, dilation=dilation,
                               stride=stride, output_padding=output_padding), # (32, 32)
            act_fn,
            nn.ConvTranspose1d(out_channels, in_channels, kernel_size, padding=padding, dilation=dilation), 
            act_fn
        )

    def forward(self, x):
        output = self.conv(x)
        return output


#  defining autoencoder
class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.encoder.to(device)

        self.decoder = decoder
        self.decoder.to(device)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
# =========================== AUTOENCODER ====================================


def calc_num_params(model, verbose=False):
    num_params = 0
    for child_name, child in model.named_children():
        if verbose: print(child_name)
        for name, param in child.named_parameters():
            if verbose: print(name)
            num_params += np.prod(np.array(param.shape))
        if verbose: print("="*50)
    return num_params