import seaborn as sns
import matplotlib.pyplot as plt


def plot_single( image, axis= None, figsize= None):
    if axis is None:
        if figsize is None:
            f, axis = plt.subplots()
        else:
            f, axis = plt.subplots( figsize= figsize)
    axis.imshow( image)
    axis.set_axis_off()
    return axis

def plot_side_by_side( image, image2, title= None):
    f, ax = plt.subplots( ncols=2)
    plot_single( image, ax[0])
    plot_single( image2, ax[1])
    if title is not None:
        if type(title)==list:
            ax[0].set_title( title[0])
            ax[1].set_title( title[1])
        else:
            ax[0].set_title( title)
            ax[1].set_title( 'Downsampled')

def plot_bbox( x_ll, y_ll, x_ru, y_ru, ax= None):
    # NOTE: this y-first signature is because numpy PIL images have x / 1st coord on the vertical
    # axis, but data annotation was with horizontal as x / 1st coord
    if ax is None:
        _, ax = plt.subplots()
    ax.plot( 
        [x_ll, x_ru, x_ru, x_ll, x_ll],[ y_ll, y_ll, y_ru, y_ru, y_ll], linewidth= 1, color= 'magenta'
    )
    
def to_magenta( arr, from_pix, to_pix, at_pix, axis):
    """
    Change a line of selected pixels to magenta (assuming a numpy array x by y by 3)
    """
    if axis== 'x':
        arr[from_pix:to_pix,at_pix,0] = 255
        arr[from_pix:to_pix,at_pix,1] = 0
        arr[from_pix:to_pix,at_pix,2] = 255
    elif axis=='y':
        arr[at_pix,from_pix:to_pix,0] = 255
        arr[at_pix,from_pix:to_pix,1] = 0
        arr[at_pix,from_pix:to_pix,2] = 255        
    return arr

def draw_bbox( arr, bbox_in_context):
    """
    Draw a bounding box in magenta for display in PIL for annotation.
    """
    x1, y1, x2, y2 = bbox_in_context
    crop = to_magenta( arr, x1, x2, y1, 'x')
    crop = to_magenta( arr, y1, y2, x2, 'y')
    crop = to_magenta( arr, x1, x2, y2, 'x')
    crop = to_magenta( arr, y1, y2, x1, 'y')
    return crop

def mark_c( crop):
    x1, y1, x2, y2 = (30, 20, 45, 35)
    crop = to_magenta( crop, x1, x2, y1, 'x')
    crop = to_magenta( crop, y1, y2, x1, 'y')
    crop = to_magenta( crop, y1, y2, x2, 'y')
    return crop

def mark_b( crop):
    x1, y1, x2, y2 = (60, 20, 75, 33)
    crop = to_magenta( crop, x1, x2, y1, 'x')
    crop = to_magenta( crop, x1+1, x2-1, y2, 'x')
    crop = to_magenta( crop, y1 + 5, y2, x1 + int((x2-x1)/2), 'y')
    crop = to_magenta( crop, y1, y2-2, x1, 'y')
    crop = to_magenta( crop, y1, y2-2, x2, 'y')
    return crop