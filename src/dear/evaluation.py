import os
import torch

import src.dear.utils as sdu
import src.dear.data as sdd

from src.dear.modelling import build_model, get_batch_from_name, get_name_from_batch

global device
device = sdu.get_device()

def load_from_folder( folder, checkpoint= None):
    # need to know the configuration:
    config_path = f'{folder}config.txt'
    # and need an existing checkpoint:
    if checkpoint is None:
        checkpoints_at = [get_batch_from_name(x) for x in os.listdir( folder) if x.endswith('.sav')]
        checkpoint = get_name_from_batch( max( checkpoints_at))
    elif type(checkpoint) == int:
        checkpoint = get_name_from_batch( checkpoint)
    checkpoint_path = f'{folder}{checkpoint}'
    assert os.path.exists( checkpoint_path)
    
    # load config for initialising the architecture correctly:
    args = sdu.read_config_from( config_path)
    
    # TODO: remove hard-coding!!!
    # TODO: make an option for no causal prior
    A_init = torch.load( '../data/xView_0/A_T.pt')
    model, _ = build_model( args, A_init, from_checkpoint= checkpoint_path, with_disc= False)
    return args, model

