
# ========================== IMPORTS ==================================
import torch
from torch import nn
from collections import OrderedDict
import numpy as np
# ========================== IMPORTS ==================================




# ========================== FLAT NET ==================================


class FlatNet(nn.Module):
    def __init__(
        self,
        num_blocks=1,
        num_channels_in=2,
        num_channels_mid=32,
        num_channels_fin=64,
        conv_kernel_size=3,
        conv_padding=2,
        conv_dilation=1,
        
        device=torch.device('cpu'),
    ):
        super().__init__()
        
        self.device = device
        
        self.iNorm = nn.InstanceNorm1d(num_features=num_channels_in)
        
        block_dict = OrderedDict([
            ('block1', ConvBlock(
            in_channels=num_channels_in,
            out_channels=num_channels_mid,
            dilation=conv_dilation,
            padding=conv_padding,
            kernel_size=conv_kernel_size))
        ])
        
        for i in range(1, num_blocks-1):
            block_dict.update([
                (f'block{i+1}', ConvBlock(
                    in_channels=num_channels_mid,
                    out_channels=num_channels_mid,
                    dilation=conv_dilation,
                    padding=conv_padding,
                    kernel_size=conv_kernel_size))
            ])
           
        block_dict.update([
            (f'block{num_blocks}', ConvBlock(
            in_channels=num_channels_mid,
            out_channels=num_channels_fin,
            dilation=conv_dilation,
            padding=conv_padding,
            kernel_size=conv_kernel_size))
        ])
        
        self.cnn = nn.Sequential(block_dict)
                
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(output_size=1)
        
        self.output_activation = nn.Sigmoid()
        
    def forward(self, _input):
        """ 
         Classify a list of samples.
        :apram input: is a list of n tensors with different height and width ...
        :return scores: tensor of scores of shape (n, #nbr_classes)
         
        """
        for i, x in enumerate(_input):
            # x is an image.
            score = self._forward(x).unsqueeze(dim=0)
            if i == 0:
                scores = score
            else:
                scores = torch.cat((scores, score), dim=0)
        return scores
    
    def _forward(self, x):
        x = x.to(self.device).float()
        x = self.iNorm(x)
        x = self.cnn(x)
        x = self.adaptive_pooling(x)
        x = torch.flatten(x)
        x = self.output_activation(x)
        return x
    
    def to_device(self, device):
        try:
            self.to(device)
            self.device = device
        except Exception as e:
            print(e)
# ========================== FLAT NET ==================================







# ======================== CONV BLOCK =================================

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=32,
        kernel_size=5,
        stride=1,
        padding=1,
        dilation=1,
        maxpool_kernel_size=2,
        drouput_p=0.2,
    ):
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        
        self.dropout = nn.Dropout1d(p=drouput_p)
        
        
        self.activation = nn.LeakyReLU()
        
        self.pooling = nn.MaxPool1d(maxpool_kernel_size)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.pooling(x)
        return x
# ======================== CONV BLOCK =================================


def calc_num_params(model, verbose=False):
    num_params = 0
    for child_name, child in model.named_children():
        if verbose: print(child_name)
        for name, param in child.named_parameters():
            if verbose: print(name)
            num_params += np.prod(np.array(param.shape))
        if verbose: print("="*50)
    return num_params