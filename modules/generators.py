from scipy import signal
import numpy as np
import torch
from torch import nn

MIN_SAMPLE_RATE = 10000
MAX_SAMPLE_RATE = 20000

MIN_DURATION = 0.1  # at least 50 samples at MIN_SAMPLE_RATE
MAX_DURATION = 10   # at most 100,000 samples at MAX_SAMPLE_RATE

MIN_BW_SWEEP = 10
MAX_BW_SWEEP = 1000

MIN_BIT_RATE = 50
MAX_BIT_RATE = 1000

MIN_CENTER_FREQ = -5000
MAX_CENTER_FREQ = 5000

MIN_PRI = 0.1
MAX_PRI = 5

MIN_MULTITONE_SPACING = 200
MIN_MULTITONE_SPACING = 1000

MIN_NUM_MULTITONES = 2
MAX_NUM_MULTITONES = 5

MIN_DUTY_CYCLE = 0.1
MAX_DUTY_CYCLE = 0.9

# TODO: make sure all params are fine and don't produce anomalies: BW too big for sample rate and creates weird stuff. On the other side don't create overfit

class SinSweepMultiToneGen(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        fs = np.random.uniform(MIN_SAMPLE_RATE, MAX_SAMPLE_RATE)
        duration = np.random.uniform(MIN_DURATION, MAX_DURATION)
        bw = np.random.uniform(MIN_BW_SWEEP, MAX_BW_SWEEP)
        pri = np.random.uniform(MIN_PRI, MAX_PRI)
        fc = np.random.uniform(MIN_CENTER_FREQ, MAX_CENTER_FREQ)
        multitone_bw = np.random.uniform(MIN_MULTITONE_SPACING, MIN_MULTITONE_SPACING)
        num_multitones = np.random.randint(MIN_NUM_MULTITONES, MAX_NUM_MULTITONES)
        
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
            raise ValueError(f"--in SinSweepMultiToneGen: Simulated 0 size signal with parameters: sample_rate: {fs}, duration: {duration}")
        t = np.linspace(0, duration, N)
        freq_shifts = np.linspace(-multitone_bw/2, multitone_bw/2 ,num_multitones)
        changing_f = fc + bw*np.sin(2*np.pi*pri*t)
        x_complex = np.vstack(
            [np.exp(1j*2*np.pi*changing_f) * np.exp(1j*2*np.pi*freq_shift) 
             for freq_shift in freq_shifts]).sum(axis=0)
        x_complex = torch.tensor(x_complex)
        return x_complex

class SinSweepGen(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        fs = np.random.uniform(MIN_SAMPLE_RATE, MAX_SAMPLE_RATE)
        duration = np.random.uniform(MIN_DURATION, MAX_DURATION)
        bw = np.random.uniform(MIN_BW_SWEEP, MAX_BW_SWEEP)
        pri = np.random.uniform(MIN_PRI, MAX_PRI)
        fc = np.random.uniform(MIN_CENTER_FREQ, MAX_CENTER_FREQ)
        
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
            raise ValueError(f"--in SinSweepGen: Simulated 0 size signal with parameters: sample_rate: {fs}, duration: {duration}")
        t = np.linspace(0, duration, N)
        changing_f = fc + bw*np.sin(2*np.pi*pri*t)
        x_complex = np.exp(1j*2*np.pi*changing_f)
        x_complex = torch.tensor(x_complex)
        return x_complex
    
class WeirdGen(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        fs = np.random.uniform(MIN_SAMPLE_RATE, MAX_SAMPLE_RATE)
        duration = np.random.uniform(MIN_DURATION, MAX_DURATION)
        duty_cycle = np.random.uniform(MIN_DUTY_CYCLE, MAX_DUTY_CYCLE)
        bw = np.random.uniform(MIN_BW_SWEEP, MAX_BW_SWEEP)
        pri = np.random.uniform(MIN_PRI, MAX_PRI)
        fc = np.random.uniform(MIN_CENTER_FREQ, MAX_CENTER_FREQ)


        return self.weird_sig(
            fs=fs,
            duration=duration,
            bw=bw,
            pri=pri,
            fc=fc,
            duty_cycle=duty_cycle
        )
    
    @staticmethod   
    def weird_sig(
        fs,
        duration,
        bw,
        pri,
        fc,
        duty_cycle
    ):
        N = int(fs*duration)
        if N == 0:
            raise ValueError(f"--in WeirdGen: Simulated 0 size signal with parameters: sample_rate: {fs}, duration: {duration}")
        t = np.linspace(0, duration, N)
        changing_f = fc + bw*signal.sawtooth(2*np.pi*pri*t, width=duty_cycle)
        x_complex = np.exp(1j*2*np.pi*np.cumsum(changing_f))
        x_complex = torch.tensor(x_complex)
        return x_complex

class LFMSweepGen(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        fs = np.random.uniform(MIN_SAMPLE_RATE, MAX_SAMPLE_RATE)
        duration = np.random.uniform(MIN_DURATION, MAX_DURATION)
        bw = np.random.uniform(MIN_BW_SWEEP, MAX_BW_SWEEP)
        pri = np.random.uniform(MIN_PRI, MAX_PRI)
        fc = np.random.uniform(MIN_CENTER_FREQ, MAX_CENTER_FREQ)
        duty_cycle = np.random.uniform(MIN_DUTY_CYCLE, MAX_DUTY_CYCLE)
        
        
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
            raise ValueError(f"--in LFMSweepGen: Simulated 0 size signal with parameters: sample_rate: {fs}, PRI: {pri}")
        t = np.linspace(0, duration, N)
        
        N_one_pulse = int(fs*pri)
        t_one_pulse = np.linspace(-0.5*pri, 0.5*pri, N_one_pulse)
        
        num_pulses = int(N / N_one_pulse) + 1
        phi_one_pulse = bw*2*np.pi*t_one_pulse**2
        phi = np.concatenate([
            phi_one_pulse * (-1)**idx + phi_one_pulse.max()*(idx%2) for idx in range(num_pulses)
        ])[:N]
        x_complex = np.exp(1j*2*np.pi*fc*t) * np.exp(1j*phi)
        x_complex = torch.tensor(x_complex)
        return x_complex
    
class BPSKGen(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        fs = np.random.uniform(MIN_SAMPLE_RATE, MAX_SAMPLE_RATE)
        duration = np.random.uniform(MIN_DURATION, MAX_DURATION)
        fc = np.random.uniform(MIN_CENTER_FREQ, MAX_CENTER_FREQ)
        bit_rate  = np.random.uniform(MIN_BIT_RATE, fs)
        
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
            raise ValueError(f"--in BPSKGen: Simulated 0 size signal with parameters: samples_per_bit: {samples_per_bit}, num_bits: {num_bits}")
        t = np.linspace(0, duration, N)
        bits = np.random.randint(0, 2, num_bits)
        x_bpsk = np.repeat(bits, samples_per_bit) - 0.5
        
        x_bpsk = torch.tensor(x_bpsk)
        return x_bpsk