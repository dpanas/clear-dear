import os
import torch

import time as tm
import pandas as pd

from functools import partial

import src.data_utils as dut
import src.dear.utils as sdu
import src.dear.data as sdd

from src.dear.modelling import build_model, get_batch_from_name, get_name_from_batch

global device
device = sdu.get_device()

to_di = lambda x: dict( [y.split(':') for y in x.split('\t')[-1].split(',')])

rename_di = {
    'D loss' : 'disc',
    ' Encoder loss' : 'enc',
    ' Decoder loss' : 'dec',
    ' Sup loss' : 'sup',
    ' E_score' : 'enc_',
    ' D score' : 'dec_'
}

def list_checkpoints( folder):
    return [
        x for x in os.listdir( folder) if x.endswith('.sav')
    ]

def parse_log( log_path):
    log_lines = dut.to_from_text( log_path).split('\n')
    df = pd.DataFrame([to_di(y) for y in log_lines[:-1]]).astype(float)
    df.rename( columns= rename_di, inplace= True)
    return df

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

def process_func( model, images, names, labels_ordered):
    z_encoded = model.encode( images)
    ims_decoded = model.decoder( z_encoded)
    df = pd.DataFrame( z_encoded[:,:3].detach().cpu().numpy(), index= names, columns= labels_ordered)
    df['rmse_recon'] = torch.sqrt( ( ( ims_decoded.detach() - images.detach() )**2 ).mean( axis= [1,2,3]) ).cpu().numpy()
    return df

def process_in_batches( data_loader, func_, labels_ordered, device= device, batch_no= None):
    df = pd.DataFrame()
    for batch_idx, (images, labels, names) in enumerate(data_loader):
        images = images.to(device)
        df = pd.concat( [df, func_( images, names, labels_ordered)])
        if batch_no is not None and batch_idx > batch_no:
            return df
    return df

def compare_from_folder( folder, labels_ordered, test_label_file= None, which= None):
    checkpoints = list_checkpoints( folder)
    checkpoints_at = [get_batch_from_name( x) for x in checkpoints]
    # need to know the configuration:
    config_path = f'{folder}config.txt'
    # load config for initialising the architecture correctly:
    args = sdu.read_config_from( config_path)
    if test_label_file is not None:
        args.label_file = test_label_file
    args.add_flips= False
    data_loader, _ = sdd.make_dataloader( args, test_flag= True)
    if which is None:
        which = [min(checkpoints_at),max(checkpoints_at)]
    elif which== 'all':
        which = checkpoints_at
    for ii in checkpoints_at:
        if ii in which:
            _, model = load_from_folder( folder, checkpoint= ii)
            model.to( device)
            process_func_ = partial( process_func, model)
            start = tm.time()
            df = process_in_batches( data_loader, process_func_, labels_ordered)
            print(f'Took {(tm.time()-start)/60} mins.')
            out_path = f'{folder}perf_model{ii}.csv'
            df.to_csv( out_path)
            del df
            print(f'Saved to {out_path}')
    