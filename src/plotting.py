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