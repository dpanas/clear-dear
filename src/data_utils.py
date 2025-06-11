import json
import pickle

import numpy as np

from PIL import Image

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