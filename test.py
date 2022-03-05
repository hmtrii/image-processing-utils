from image_utils import *
from box_utils import IoU


if __name__ == '__main__':
    box2 = [0, 0, 10, 10]
    box1 = [9, 9, 20, 20]
    # print(overlap_area(box1, box2))
    print(IoU(box1, box2))
    print()