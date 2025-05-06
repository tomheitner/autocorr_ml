from scipy import signal
import torch
from torch import nn
import numpy as np


class AutoCorrPeak(nn.Module):
    def __init__(
        self,
        peak_width_factor=10,
        autocorr_mode='same',
    ):
        super().__init__()
        self.peak_width_factor = peak_width_factor
        self.autocorr_mode = autocorr_mode
        
    def forward(self, x):
        autocorr_sig = signal.correlate(x, x, mode=self.autocorr_mode)

        '''
        Amplitude
        '''  
        abs_acorr = abs(autocorr_sig)

        peak_loc = len(abs_acorr) // 2

        peaks_width, _, _, _ = signal.peak_widths(abs_acorr, [peak_loc])
        peak_width = peaks_width[0]

        '''
        PHASE
        '''
        phase_unwarp = np.unwrap(np.angle(autocorr_sig))



        '''
        PEAK ENVIROMENTS
        '''
        peak_width*=self.peak_width_factor
        peak_width = np.clip(peak_width, a_min=5, a_max=len(abs_acorr))
        peak_abs_env = abs_acorr[int(peak_loc-peak_width//2) : int(peak_loc+peak_width//2)]
        peak_phase_env = phase_unwarp[int(peak_loc-peak_width//2) : int(peak_loc+peak_width//2)]

        peak_env = torch.stack((
            torch.tensor(peak_abs_env),
            torch.tensor(peak_phase_env)
        ))

        return peak_env