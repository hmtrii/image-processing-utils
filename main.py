import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from PIL import Image
import numpy as np


def show_images(images, n_rows, n_columns, fig_size=None, titles=[], title_sizes=14, save_file='', sup_title='', sup_title_size=20):
    """ Plot multiple numpy images

    Args:
        images (list): The numpy images
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

def plot_boxes(image, boxes, color=(255, 0, 0), thickness=2):
    """ Plot boxes into the image

    Args:
        image (numpy array): the drawing image
        boxes (list): drawing boxes. Each box is a list containing [xmin, ymin, xmax, ymax]
    """
    
    if np.array(boxes).ndim == 1:
        boxes = [boxes]
    if isinstance(thickness, int):
        thickness = [thickness] * len(boxes)
    else:
        assert len(boxes) == len(thickness), print(f'Assert length of boxes and thickness is the same, len(boxes) = {len(boxes)} and len(color) = {len(thickness)}')
    if np.array(color).ndim == 1:
        color = [color] * len(boxes)
    else:
        assert len(boxes) == len(color), print(f'Assert length of boxes and color is the same, len(boxes) = {len(boxes)} and len(color) = {len(color)}')

    for box, c, t in zip(boxes, color, thickness):
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[2]), color=c, thickness=t)


if __name__ == '__main__':
    # image = cv2.imread('./test-images/puppy.jpg')
    # boxes = [[100, 120, 600, 500], [1000, 1000, 1200, 1300], [200, 300, 700, 1000]]
    # colors = [(255, 0, 0), [0, 255, 0], [0, 0, 255]]
    # thickness = [20, 2, 3]
    # plot_boxes(image, boxes, color=colors, thickness=thickness)
    # show_images([image], 1, 1, save_file='test-output/test.png')
    print()