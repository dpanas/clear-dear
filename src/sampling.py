import sys

import numpy as np
import pandas as pd

from .geometry import center_on_bbox, square_bbox_around

def sample_xy( image_dims, num_samp, res= 128, seed= 11):
    """
    Sample x and y coordinates to take a res by res square crop around it (assuming
    large satellite image tiles).
    """
    # NOTE: sampling is in numpy coordinates of an image, with x being vertical axis!
    res_half = int( res/2)
    xmax, ymax = image_dims
    np.random.seed( seed)
    # sample valid x and y coordinates (at least half-resolution away from edge):
    xs = res_half + np.random.uniform( xmax - res, size= num_samp)
    ys = res_half + np.random.uniform( ymax - res, size= num_samp)
    return pd.DataFrame( {'xs': xs, 'ys': ys}).sort_values( by= 'xs').astype(int)

def filter_xy( dfs, shift_min, rounds= 2, verbose= False):
    """
    Roughly filter the x,y pairs to remove near-duplicates, a.k.a. locations that are v close.
    """
    # we don't want to sample points that are too close:    
    for rd in range( rounds):        
        to_check = np.argwhere( abs( ( dfs['xs'] - np.roll( dfs['xs'], shift= 1))) < shift_min).reshape(-1)
        to_drop = []
        for index in to_check:
            if abs( dfs.iloc[[index-1,index]]['ys'].diff().values[-1]) < shift_min:
                p1, p2 = dfs['roads'].iloc[[index-1,index]]
                if p1 < p2:
                    to_drop.append( dfs.iloc[[index-1]].index[0])
                else:
                    to_drop.append( dfs.iloc[[index]].index[0])
                if verbose:
                    print(f'Round {rd+1}, removing one as too close:\n{dfs[['xs','ys','roads']].iloc[[index-1,index]]}')
                
        dfs.drop( labels= to_drop, axis= 0, inplace= True)
    return dfs

def name_sample_image( source_image_id, x, y):
    """
    Define a naming convention for the sampled images.
    """
    return f'tif{source_image_id}_x{str(x)}_y{str(y)}.png'

def prep_samples_unlabelled( 
    image_dims, image_name, num_samp, res, seed, shift_min= None, rounds= 2, label_names= ['buildings','roads','cars'],
    verbose= False, df= None
):
    """
    Sample x,y coordinates and filter out ones that may be too near to one another. Since these are
    to be unlabelled samples (and `dear` marks unlabelled with a -1), add the desired label columns,
    also will need image name and the numpy bbox.
    """
    shift_min = shift_min if shift_min is not None else int( res/ 5)
    if df is None:
        df = sample_xy( image_dims, num_samp, res, seed)
    # set labels to -1 
    df[ label_names] = -1, -1, -1
    df = filter_xy( df, shift_min, rounds)
    # now add the target (numpy) bbox:
    df['bbox_np'] = df.apply(
        lambda x: square_bbox_around( x['xs'], x['ys'], res= res), axis= 1
    ).map(
        lambda x: [int(y) for y in x]
    )
    df['image'] = df.apply( lambda x: name_sample_image( image_name, x['xs'], x['ys']), axis= 1)
    return df

def prep_samples_for_annot( 
    image_dims, image_name, num_samp, res, seed, shift_min= None, rounds= 2, label_names= ['buildings','roads','cars'],
    verbose= False, df= None
):
    """
    Given an image (as a numpy array), sample random x,y coordinates for res by res crops, minding
    not to be too close to the edge or each other (within reason)
    """
    xmax, ymax = image_dims
    dfs = prep_samples_unlabelled(
        image_dims, image_name, num_samp, res, seed, shift_min, rounds, label_names, df= df
    )
    # for annotation, let's add a 'context' bounding box, just to have visual aid:
    dfs['bbox_ctx'] = dfs.apply( 
        lambda x: center_on_bbox( x['bbox_np'], 256, xmax, ymax, fix_edge= True), axis= 1
    )
    dfs['bbox_context'] = dfs['bbox_ctx'].apply( lambda x: x[0]).map(lambda x: [int(y) for y in x])
    dfs['bbox_in_context'] = dfs['bbox_ctx'].apply( lambda x: x[1]).map(lambda x: [int(y) for y in x])
    # prepare space for pre-annotating (from the xView bounding boxes for cars and buildings)
    dfs['built_area'] = np.nan
    dfs['n_cars'] = np.nan
    dfs['n_bus_trucks'] = np.nan
    return dfs.sort_values(by=['xs','ys']).reset_index(drop=True).drop( labels= ['bbox_ctx'], axis=1) 