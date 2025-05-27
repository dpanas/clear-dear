"""Geometric utilitites for working with xView bounding boxes"""

def get_dims( bbox):
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

def bbox_np( bbox):
    return bbox[1], bbox[0], bbox[3], bbox[2]

def center_on_bbox( bbox, res, xmax, ymax, fix_edge= False):
    """
    Assuming an image of (xmax,ymax) pixels, and a bounding box within the picture,
    return coordinates of a square centered on the center of bbox, with res width and height.
    
    :param bbox: quad-tuple of x and y coordinates, x1, y1, x2, y2, 1= lower left, 2= upper right
    :param res: pixel resolution of requested 'meta' bounding box
    :param xmax: dimension along x-axis in pixels
    :param ymax: dimension along y-axis in pixels
    :return bbox_meta: the bbox-surrounding res by res 'meta' bounding box; None if it goes beyond image border
    :return bbox_relative: original bounding box relative to the 
    """
    x1, y1, x2, y2 = bbox 
    x, y = x1 + int( (x2 - x1)/2), y1 + int( (y2 - y1)/2)    
    x1_, x2_ = x - int( res/2), x + int(res/2)    
    y1_, y2_ = y - int( res/2), y + int(res/2)    
    bbox_meta = [x1_, y1_, x2_, y2_]
    bbox_relative = [x1 - x1_, y1 - y1_, x2 - x1_, y2 - y1_]
    if fix_edge:
        if x1_ < 0:
            for bbox_ in [bbox_meta,bbox_relative]:
                for ix in [0,2]:
                    bbox_[ix] -= x1_
        if y1_ < 0:
            for bbox_ in [bbox_meta,bbox_relative]:
                for ix in [1,3]:
                    bbox_[ix] -= y1_
        if x2_ > xmax:
            for bbox_ in [bbox_meta,bbox_relative]:
                for ix in [0,2]:
                    bbox_[ix] -= x2_
        if y2_ > ymax:
            for bbox_ in [bbox_meta,bbox_relative]:
                for ix in [1,3]:
                    bbox_[ix] -= y2_
        return bbox_meta, bbox_relative
    else:
        try:
            assert x1_ >= 0 and x2_ <= xmax and y1_ >=0 and y2_ <= ymax
            return bbox_meta, bbox_relative
        except AssertionError:
            return None, None
    
def crop_to_bbox( image_array, bbox):
    x1, y1, x2, y2 = bbox
    return image_array[x1:x2,:,:][:,y1:y2,:]

def intersect( bbox_a, bbox_b):
    x1_a, y1_a, x2_a, y2_a = bbox_a
    x1_b, y1_b, x2_b, y2_b = bbox_b
    x1_cond = (x1_a < x1_b < x2_a)
    x2_cond = (x1_a < x2_b < x2_a)
    y1_cond = (y1_a < y1_b < y2_a)
    y2_cond = (y1_a < y2_b < y2_a)
    #print(x1_cond,x2_cond)
    return ( x1_cond or x2_cond ) and (  y1_cond or y2_cond )