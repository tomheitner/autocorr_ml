from scipy import signal
import torch
from torch import nn
import numpy as np

import torch
import numpy as np
from torch import nn

class SinSweepMultiToneGen(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        fs = np.random.uniform(100, 10000)
        duration = np.random.uniform(0.1, 10)
        bw = np.random.uniform(100, 200)
        pri = np.random.uniform(0.01, 5)
        fc = np.random.uniform(100, 8000)
        multitone_bw = np.random.uniform(500, 8000)
        num_multitones = np.random.randint(2, 10)
        
        return self.sin_sweep(
            fs=fs,
            duration=duration,
            bw=bw,
            pri=pri,
            fc=fc,
            multitone_bw=multitone_bw,
            num_multitones=num_multitones,
        )
    
    @staticmethod   
    def sin_sweep(
        fs,
        duration,
        bw,
        pri,
        fc,
        multitone_bw,
        num_multitones
    ):
        N = int(fs*duration)
        if N == 0:
            print(fs, duration)
            raise
        t = np.linspace(0, duration, N)
        freq_shifts = np.linspace(-multitone_bw/2, multitone_bw/2 ,num_multitones)
        changing_f = fc + bw*np.sin(2*np.pi*pri*t)
        x_complex = np.array(
            [np.exp(1j*2*np.pi*changing_f) * np.exp(1j*2*np.pi*freq_shift) 
             for freq_shift in freq_shifts]).sum(axis=0)
        x_complex = torch.tensor(x_complex)
        return x_complex

class SinSweepGen(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        fs = np.random.uniform(100, 10000)
        duration = np.random.uniform(0.1, 10)
        bw = np.random.uniform(100, 8000)
        pri = np.random.uniform(0.01, 5)
        fc = np.random.uniform(100, 8000)
        
        return self.sin_sweep(
            fs=fs,
            duration=duration,
            bw=bw,
            pri=pri,
            fc=fc,
        )
    
    @staticmethod   
    def sin_sweep(
        fs,
        duration,
        bw,
        pri,
        fc

    ):
        N = int(fs*duration)
        if N == 0:
            print(fs, duration)
            raise
        t = np.linspace(0, duration, N)
        changing_f = fc + bw*np.sin(2*np.pi*pri*t)
        x_complex = np.exp(1j*2*np.pi*changing_f)
        x_complex = torch.tensor(x_complex)
        return x_complex
    
class LFMSweepGen(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        fs = np.random.uniform(100, 10000)
        duration = np.random.uniform(0.1, 10)
        bw = np.random.uniform(100, 8000)
        pri = np.random.uniform(0.01, 5)
        fc = np.random.uniform(100, 8000)
        duty_cycle = np.random.uniform(0.1, 0.9)
        
        return self.lfm_sweep(
            fs=fs,
            duration=duration,
            bw=bw,
            pri=pri,
            fc=fc,
            duty_cycle=duty_cycle
        )
    
    @staticmethod   
    def lfm_sweep(
        fs,
        duration,
        bw,
        pri,
        fc,
        duty_cycle
    ):
        N = int(fs*duration)
        if N == 0:
            print(fs, duration)
            raise
        t = np.linspace(0, duration, N)
        changing_f = fc + bw*signal.sawtooth(2*np.pi*pri*t, width=duty_cycle)
        x_complex = np.exp(1j*2*np.pi*changing_f)
        x_complex = torch.tensor(x_complex)
        return x_complex
    
class BPSKGen(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        fs = np.random.uniform(100, 10000)
        duration = np.random.uniform(0.1, 10)
        fc = np.random.uniform(100, 8000)
        bit_rate  = np.random.uniform(10, fs)
        
        return self.bpsk(fs=fs, duration=duration, fc=fc, bit_rate=bit_rate)
    
    @staticmethod   
    def bpsk(
        fs,
        duration,
        fc,
        bit_rate

    ):
        samples_per_bit = int(np.round(fs / bit_rate))
        num_bits = int(bit_rate * duration)
        N = samples_per_bit * num_bits
        if N == 0:
            print(samples_per_bit, num_bits)
            raise
        t = np.linspace(0, duration, N)
        bits = np.random.randint(0, 2, num_bits)
        x_bpsk = np.repeat(bits, samples_per_bit) - 0.5
        
        x_bpsk = torch.tensor(x_bpsk)
        return x_bpsk