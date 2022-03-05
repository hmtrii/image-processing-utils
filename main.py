from image_utils import *
from box_utils import *


if __name__ == '__main__':
    image = cv2.imread('./test-images/puppy.jpg')
    boxes = [[100, 120, 600, 500], [1000, 1000, 1200, 1300], [200, 300, 700, 1000]]
    colors = [(255, 0, 0), [0, 255, 0], [0, 0, 255]]
    thickness = [20, 2, 3]
    # texts = ['abc' ,'efg', 'qwe']
    # texts = ''
    plot_boxes(image, boxes, color=colors, thickness=thickness)
    show_images([image], 1, 1, save_file='test-output/test.png')
    print()