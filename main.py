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

def plot_boxes(images, boxes):
    return


if __name__ == '__main__':
    # image1 = cv2.imread('test-images/GiBehold.png')
    # image1 = Image.open('test-images/GiBehold.png')
    # image1 = np.array(image1)
    # image2 = cv2.imread('test-images/GiPoliceOfficerHead.png')
    # image3 = cv2.imread('test-images/GiWaterTank.png')
    # image4 = cv2.imread('test-images/ImInfinite.png')
    # show_images([image1, image2], 1, 2, \
    #     save_file='test-output/test.png', sup_title='aaa',\
    #     titles=['q', 'w'], fig_size=(20, 20), title_sizes=50)
    print()