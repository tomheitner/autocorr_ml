import numpy as np
import torch

class SignalDataset:
    def __init__(self, gen_objs, transform=None):
        np.random.seed(999)
        torch.random.manual_seed(555)
        
        self.gen_objs = gen_objs
        self.transform = transform
    
    def __iter__(self):
        return self
    
    def __next__(self):
        gen_obj_idx = np.random.randint(0, len(self.gen_objs))
        gen_obj = self.gen_objs[gen_obj_idx]
        gen_signal = gen_obj()  # generate
        
        if self.transform is not None:
            peak1 = self.transform(gen_signal)
            peak2 = self.transform(gen_signal)
            
            return peak1, peak2, gen_obj_idx
        
        else: 
            print(f"shouldn't get here {gen_obj}")
            raise
            # return gen_signal