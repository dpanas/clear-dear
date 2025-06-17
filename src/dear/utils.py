import os
import torch

import numpy as np
import pandas as pd

from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

class DummyArgs():
    def __init__( self, di):
        self.__dict__.update( **di)

def draw_recon(x, x_recon):
    x_l, x_recon_l = x.tolist(), x_recon.tolist()
    result = [None] * (len(x_l) + len(x_recon_l))
    result[::2] = x_l
    result[1::2] = x_recon_l
    return torch.FloatTensor(result)

def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def scale_output( x):
    x_ = x.detach().numpy()
    x_ = x_ - x_.min()
    x_ = x_/ x_.max()
    return (x_ * 255).astype(np.uint8)

def write_config_to_file(config, save_path):
    with open(os.path.join(save_path, 'config.txt'), 'w') as file:
        for arg in vars(config):
            file.write(str(arg) + ': ' + str(getattr(config, arg)) + '\n')

def get_device( sagan_obj= None):
    if sagan_obj is not None:
        if not sagan_obj.config.disable_cuda and torch.cuda.is_available():
            print("CUDA is available!")
            sagan_obj.device = torch.device('cuda')
            sagan_obj.config.dataloader_args['pin_memory'] = True
        else:
            print("Cuda is NOT available, running on CPU.")
            sagan_obj.device = torch.device('cpu')
        if torch.cuda.is_available() and sagan_obj.config.disable_cuda:
            print("WARNING: You have a CUDA device, so you should probably run without --disable_cuda")
    else:
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

