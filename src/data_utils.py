import json
import pickle

import numpy as np
import pandas as pd

from PIL import Image
import src.geometry as sgeo 


def from_json( file_path, encoding= None):
    with open( file_path, 'r', encoding= encoding) as inn:
        dictionary = json.load( inn)
    return dictionary 

def load_image( path):
    return Image.open( path).convert( 'RGB')

def to_numpy( image_tensor):
    """
    Convenience util for analysis of single images when read by PIL or torch/vision.
    """
    image_array = np.asarray( image_tensor)
    if image_array.max() < 1:
        image_array = (image_array * 255).astype(np.uint8)
    
    if image_array.shape[0] == 3:
        return np.swapaxes( np.swapaxes( image_array, 0, 1), 1, 2)
    elif image_array.shape[-1] == 3:
        return image_array
    else:
        raise Exception(f'Weird shape for an image: {image_array.shape}')

def to_from_pickle( file_path, content= None):
    mode = 'wb' if content is not None else 'rb'
    with open( file_path, mode) as stream:
        if mode.startswith('r'):
            return pickle.load( stream)
        pickle.dump( content, stream)
        
def to_from_text( file_path, content= None, encoding= 'utf-8', as_bytes= False):
    mode = 'w' if content is not None else 'r'
    if as_bytes:
        mode += 'b'
        encoding = None
    with open( file_path, mode, encoding= encoding) as stream:
        if mode.startswith('r'):
            return stream.read()
        stream.write( content)

def bbox_from_str( bbox_str):
    return [int(y.strip()) for y in bbox_str.strip('[]()').split(',')]

## --- xView specific stuff:

xView_car_set = set([17,18,20])
xView_bus_truck_set = set([19,21,23,24,25,26,27,28])

xView_building_flagger = lambda x: x == 73
xView_car_flagger = lambda x: x in xView_car_set
xView_bus_truck_flagger = lambda x: x in xView_bus_truck_set

def parse_xView_geojson( file_path, labelname_path):
    label_li = from_json( file_path)['features']
    labelnames_di = to_from_pickle( labelname_path)
    df = pd.DataFrame([ x['properties'] for x in label_li])
    df['bbox'] = df['bounds_imcoords'].map( bbox_from_str)
    df.drop(
        labels= ['cat_id','edited_by','point_geom','grid_file','ingest_time','bounds_imcoords'], axis= 1, inplace= True
    )
    df.rename( columns= {'type_id':'cat_id'}, inplace= True)
    df['cat'] = df['cat_id'].map( labelnames_di)
    df['bbox_np'] = df['bbox'].apply( sgeo.bbox_np)
    df['feature_area'] = df['bbox'].apply( sgeo.get_area)
    df['building_flag'] = df['cat_id'].map( xView_building_flagger).astype(int)
    df['car_flag'] = df['cat_id'].map( xView_car_flagger).astype(int)
    df['bus_truck_flag'] = df['cat_id'].map( xView_bus_truck_flagger).astype(int)
    return df