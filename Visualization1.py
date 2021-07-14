import cv2
import numpy as np
from geoplotlib.utils import read_csv
from skimage import io, data
import matplotlib.pyplot as plt
import geoplotlib

def four_nn_label_map(img):
    """

    :param img:
    :return:
    """
    height, width = img.shape
    label_map_d = {}
    label = 1
    label_map = np.zeros((height,width))
    for row in range(height):
        for col in range(width):
            if (img[row,col] == 255).all():

                left_p = label_map[max(row-1,0),col]
                up_p = label_map[row,max(col-1,0)]
                temp_l = max(left_p,up_p)
                if temp_l != 0:
                    label_map[row,col] = temp_l
                    label_map_d[temp_l].append((row, col))
                else:
                    label_map[row, col] = label
                    label_map_d[label] = [(row,col)]
                    label += 1
    delete_l = []
    for ele in label_map_d:
        if len(label_map_d[ele]) <2:
            delete_l.append(ele)
    for ele in delete_l:
        del label_map_d[ele]
    return label_map_d

def binary_map_create(img):
    """

    :param img:
    :return:
    """
    rows, cols = img.shape
    binary_img = np.zeros((rows,cols))
    for row in range(rows):
        for col in range(cols):
            if (img[row,col] == 255).all():
                binary_img[row,col] = np.int16(255)
    return binary_img

def neighbor_value(img, label_d, binary_img: np.array, offsets, reverse=False):
    rows, cols = binary_img.shape
    label_idx = 0
    rows_ = [0, rows, 1] if reverse == False else [rows-1, -1, -1]
    cols_ = [0, cols, 1] if reverse == False else [cols-1, -1, -1]
    for row in range(rows_[0], rows_[1], rows_[2]):
        for col in range(cols_[0], cols_[1], cols_[2]):
            label = 256
            if (img[row][col] != 255).all():
                continue
            for offset in offsets:
                neighbor_row = min(max(0, row + offset[0]), rows - 1)
                neighbor_col = min(max(0, col + offset[1]), cols - 1)
                neighbor_val = img[neighbor_row, neighbor_col]
                if (neighbor_val != 255).all():
                    continue
                neighbor_label = binary_img[neighbor_row,neighbor_col]
                label = neighbor_label if neighbor_label < label else label
            if label == 255:
                label_idx += 1
                label = label_idx
            if label not in label_d:
                label_d[label] = [(row,col)]
            elif (row,col) not in label_d[label]:
                label_d[label].append((row,col))
            if binary_img[row,col] != label and int(binary_img[row,col] in label_d):
                del_index = label_d[int(binary_img[row,col])].index((row,col))
                label_d[int(binary_img[row,col])].pop(del_index)
            binary_img[row][col] = label
    return binary_img, label_d

def Two_Pass(img: np.array):
    offsets = [[0, -1], [-1, 0], [0, 0], [1, 0], [0, 1]]
    binary_img = binary_map_create(img)
    label_d = {}
    binary_img,label_d = neighbor_value(img, label_d, binary_img, offsets, False)
    binary_img,label_d = neighbor_value(img, label_d, binary_img, offsets, True)
    label_d = label_d_fix(label_d)

    return binary_img, label_d

def label_d_fix(d):
    del_list = []
    for ele in d:
        if len(d[ele]) <2:
            del_list.append(ele)
    for ele in del_list:
        del d[ele]
    return d

image = io.imread('./taxi_zone_map_staten_island.jpg')
img = image.copy()
#print(image[478,656])
#print((image[478,656]!=[201,242,208]).all())
#print(np.where(image[:,:,0]== 201 and image[:,:,1] == 242 and image[:,:,2] == 208))
a = np.load('./staten_island_label_d.npy',allow_pickle=True)
b = dict(a.item())
img[700:720,800:820] = [1,255,255]
io.imshow(img)
plt.show()


#d={1:[(0,0),(0,3),(0,5)],2:[(1,3),(1,5)]}


#print(type(d))
#np.save('./test',d)

binary1_img = np.zeros((4, 7), dtype=np.int16)
index = [[0, 2], [0, 5],
        [1, 0], [1, 1], [1, 2], [1, 4], [1, 5], [1, 6],
        [2, 2], [2, 5],
        [3, 1], [3, 2], [3, 4], [3, 6]]
for i in index:
    binary1_img[i[0],i[1]] = np.int16(255)
#print(binary1_img)
#a_bi, d = Two_Pass(binary1_img)
#print(a_bi)
#print(d)
#io.imshow(image)
#plt.show()
#a = [[0,1],[0,2],[2,2]]

