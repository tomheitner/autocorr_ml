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
    
class XCorrPeak(nn.Module):
    def __init__(
        self,
        peak_width_factor=10,
        corr_mode='same',
    ):
        super().__init__()
        self.peak_width_factor = peak_width_factor
        self.corr_mode = corr_mode
        
    def forward(self, x, x_original):
        xcorr_sig = signal.correlate(x, x_original, mode=self.corr_mode)

        '''
        Amplitude
        '''  
        abs_xcorr = abs(xcorr_sig)

        # peak_loc = len(abs_xcorr) // 2
        peak_loc = abs_xcorr.argmax()
        
        peaks_width, _, _, _ = signal.peak_widths(abs_xcorr, [peak_loc])
        peak_width = peaks_width[0]

        '''
        PHASE
        '''
        phase_unwarp = np.unwrap(np.angle(xcorr_sig))



        '''
        PEAK ENVIROMENTS
        '''
        peak_width*=self.peak_width_factor
        peak_width = np.clip(peak_width, a_min=5, a_max=2*peak_loc)
        peak_abs_env = abs_xcorr[int(peak_loc-peak_width//2) : int(peak_loc+peak_width//2)]
        peak_phase_env = phase_unwarp[int(peak_loc-peak_width//2) : int(peak_loc+peak_width//2)]

        peak_env = torch.stack((
            torch.tensor(peak_abs_env),
            torch.tensor(peak_phase_env)
        ))
        
        any_zeros = sum([peak_env.shape[i]==0 for i in range(len(peak_env.shape))])
        if any_zeros:
            raise ValueError(f"--in XCorrPeak: peak_env shape has 0! peak_width: {peak_width}, peak_env: {int(peak_loc-peak_width//2)} to {int(peak_loc+peak_width//2)}")
        return peak_env