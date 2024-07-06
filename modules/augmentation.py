import torch
from torch import nn
import numpy as np

class AWGN(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        noise_level = abs(x).max() / abs(x).median()
        noised = x + torch.tensor(np.random.normal(0, scale=noise_level, size=len(x)))
        return noised
    
class MP(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = x.type(torch.complex128)
        num_mp = np.random.randint(1, 4)
        amp = np.clip(np.random.exponential(0.1, size=num_mp), 0, 0.4)
        phase = np.random.uniform(0, 2*np.pi, size=num_mp)
        delay = np.clip(np.random.geometric(0.1, size=4), 1, 50)
        
        for mp_comp_idx in range(num_mp): 
            x += torch.tensor(amp[mp_comp_idx]*np.roll(x, delay[mp_comp_idx]) * np.exp(1j*phase[mp_comp_idx]))
        return x
    
    
class Channel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        num_mp = np.random.randint(1, 4)
        amp = np.clip(np.random.exponential(0.1, size=num_mp), 0, 0.4)
        phase = np.random.uniform(0, 2*np.pi, size=num_mp)
        delay = np.clip(np.random.geometric(0.1, size=4), 1, 50)
        
        for mp_comp_idx in range(num_mp): 
            x += torch.tensor(amp[mp_comp_idx]*np.roll(x, delay[mp_comp_idx]) * np.exp(1j*phase[mp_comp_idx]))
        return x