"""Geometric utilitites for working with xView bounding boxes"""

def get_dims( bbox):
    """
    To asses size of objects for possible filtering / annotating help.
    """
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

def bbox_np( bbox):
    """
    Numpy / PIL have swapped axes wrt pyplot, and the bounding boxes in the data are in
    pyplot ref system, will need to move back and forth between them...
    """
    return bbox[1], bbox[0], bbox[3], bbox[2]

def square_bbox_around( x, y, res):
    x1_, x2_ = x - int( res/2), x + int(res/2)    
    y1_, y2_ = y - int( res/2), y + int(res/2)    
    return x1_, y1_, x2_, y2_

def center_on_bbox( bbox, res, xmax, ymax, fix_edge= False):
    """
    Assuming an image of (xmax,ymax) pixels, and a bounding box within the picture,
    return coordinates of a square centered on the center of bbox, with res width and height.
    
    :param bbox: quad-tuple of x and y coordinates, x1, y1, x2, y2, 1= lower left, 2= upper right
    :param res: pixel resolution of requested 'meta' bounding box
    :param xmax: dimension along x-axis in pixels
    :param ymax: dimension along y-axis in pixels
    :param fix_edge: bool to indicate whether to move the 'meta' bbox so that it doesn't go outwith image edge
    :return bbox_meta: the bbox-surrounding res by res 'meta' bounding box; None if it goes beyond image border
    :return bbox_relative: original bounding box relative to the 
    """
    x1, y1, x2, y2 = bbox 
    x, y = x1 + int( (x2 - x1)/2), y1 + int( (y2 - y1)/2)    
    x1_, y1_, x2_, y2_ = square_bbox_around( x, y, res)
    bbox_meta = [x1_, y1_, x2_, y2_]
    bbox_relative = [x1 - x1_, y1 - y1_, x2 - x1_, y2 - y1_]
    if fix_edge:
        # if the surrounding 'meta' bbox for cropping is too close to the edge, move appropriately
        if x1_ < 0:
            for ii, bbox_ in enumerate([bbox_meta,bbox_relative]):
                for ix in [0,2]:
                    bbox_[ix] -= x1_ * (-1)**ii # NOTE: the relative bbox also needs re-referenced in the opposite dir
        if y1_ < 0:
            for ii, bbox_ in enumerate([bbox_meta,bbox_relative]):
                for ix in [1,3]:
                    bbox_[ix] -= y1_ * (-1)**ii
        if x2_ > xmax:
            x2__ = x2_ - xmax
            for ii, bbox_ in enumerate([bbox_meta,bbox_relative]):
                for ix in [0,2]:
                    bbox_[ix] -= x2__ * (-1)**ii
        if y2_ > ymax:
            y2__ = y2_ - ymax
            for ii, bbox_ in enumerate([bbox_meta,bbox_relative]):
                for ix in [1,3]:
                    bbox_[ix] -= y2__ * (-1)**ii
        return bbox_meta, bbox_relative
    else:
        try:
            assert x1_ >= 0 and x2_ <= xmax and y1_ >=0 and y2_ <= ymax
            return bbox_meta, bbox_relative
        except AssertionError:
            return None, None
    
def crop_to_bbox( image_array, bbox):
    """
    Crop image to just a bbox (the numpy-version of bbox)
    """
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