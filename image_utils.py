import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from PIL import Image
import numpy as np


def show_images(images :np.array, n_rows: int=1, n_columns: int=1, fig_size: tuple=None, titles: list=[], title_sizes: int=14, save_file: str='', sup_title: str='', sup_title_size: int=20) -> None:
    """ Plot multiple numpy images

    Args:
        images (list): The list of numpy images
        n_rows (int): number of rows in the figure
        n_columns (int): number of column in the figure
        fig_size (tuple(int, int)): size of the figure
        titles (list):  title of each image in the list images respectively
        title_sizes (str, list): size of each titles respectively
        save_file (str): file to save the figure, not save if save_file is none
        sup_title (str): the title of the figure
        sup_title_size (int): size of sup_title
    """

    assert type(images) == list, print('Type of images is not a list')
    assert len(images) <= n_rows * n_columns, print('Number of image greater than size of grid')

    if not titles:
        titles = [None] * len(images)
    if isinstance(title_sizes, int):
        title_sizes = [title_sizes] * len(images)

    gs = gridspec.GridSpec(n_rows, n_columns)
    fig = plt.figure(figsize=fig_size)
    for i in range(len(images)):
        plt.subplot(gs[i])
        plt.subplot(n_rows, n_columns, i+1)
        if (len(images[i].shape) < 3):
            plt.imshow(images[i], plt.cm.gray)
        else:
            plt.imshow(images[i])
        plt.title(titles[i], fontsize=title_sizes[i])
        plt.axis('off')
    if sup_title:
        fig.suptitle(sup_title, fontsize=sup_title_size)
    plt.show()
    if save_file:
        plt.savefig(save_file)

def plot_boxes(image: np.array, box: list, color: tuple=(255, 0, 0), thickness: int=2, text :list=[]) -> np.array:
    """ Plot boxes into the image

    Args:
        image (numpy.array): the drawing image
        boxes (list): drawing boxes. Each box is a list containing [xmin, ymin, xmax, ymax]
    
    Return:
        A ploted image.
    """

    plot_image = image.copy()
    box = np.array(box, dtype=np.int32)
    if np.array(box).ndim == 1:
        box = [box]
    if isinstance(thickness, int):
        thickness = [thickness] * len(box)
    else:
        assert len(box) == len(thickness), print(f'Assert length of boxes and thickness are the same, len(boxes) = {len(box)} and len(color) = {len(thickness)}')
    if np.array(color).ndim == 1:
        color = [color] * len(box)
    else:
        assert len(box) == len(color), print(f'Assert length of boxes and color are the same, len(boxes) = {len(box)} and len(color) = {len(color)}')
    if not text:
        text = [None] * len(box)
    if isinstance(text, str):
        text = [text] * len(box)
    else:
        assert len(box) == len(text), print(f'Assert length of boxes and texts are are the same, len(boxes = {len(box)} and len(text) = {len(text)}')

    for b, c, th, te in zip(box, color, thickness, text):
        cv2.rectangle(plot_image, (b[0], b[1]), (b[2], b[3]), color=c, thickness=th)
        cv2.putText(plot_image, str(te), (b[0], b[1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=c, thickness=th)
    
    return plot_image

