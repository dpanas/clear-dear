import json

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
    if image_array.shape[0] == 3:
        return np.swapaxes( np.swapaxes( np.asarray( image_tensor), 0, 1), 1, 2)
    elif image_array.shape[-1] == 3:
        return image_array
    else:
        raise Exception(f'Weird shape for an image: {image_array.shape}')