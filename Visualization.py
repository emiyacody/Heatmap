import cv2
import numpy as np
from skimage import io, data
import matplotlib.pyplot as plt

NEIGHBOR_HOODS_4 = True
OFFSETS_4 = [[0, -1], [-1, 0], [0, 0], [1, 0], [0, 1]]

def reorganize(binary_img: np.array):
    index_map = []
    points = []
    index = -1
    rows, cols = binary_img.shape
    for row in range(rows):
        for col in range(cols):
            var = binary_img[row][col]
            if var < 0.5:
                continue
            if var in index_map:
                index = index_map.index(var)
                num = index + 1
            else:
                index = len(index_map)
                num = index + 1
                index_map.append(var)
                points.append([])
            binary_img[row][col] = num
            points[index].append([row, col])
    return binary_img, points

def recursive_seed(binary_img: np.array, seed_row, seed_col, offsets, num, max_num=100):
    rows, cols = binary_img.shape
    binary_img[seed_row][seed_col] = num
    for offset in offsets:
        neighbor_row = min(max(0, seed_row+offset[0]), rows-1)
        neighbor_col = min(max(0, seed_col+offset[1]), cols-1)
        var = binary_img[neighbor_row][neighbor_col]
        if var < max_num:
            continue
        binary_img = recursive_seed(binary_img, neighbor_row, neighbor_col, offsets, num, max_num)
    return binary_img

def Seed_Filling(binary_img, neighbor_hoods, max_num=100):
    if neighbor_hoods == NEIGHBOR_HOODS_4:
        offsets = OFFSETS_4
    else:
        raise ValueError

    num = 1
    rows, cols = binary_img.shape
    for row in range(rows):
        for col in range(cols):
            var = binary_img[row][col]
            if var <= max_num:
                continue
            binary_img = recursive_seed(binary_img, row, col, offsets, num, max_num=100)
            num += 1
    return binary_img

print("Seed_Filling")
binary_img = Seed_Filling(binary_img, NEIGHBOR_HOODS_4)
binary_img, points = reorganize(binary_img)
print(binary_img, points)

image = io.imread('./taxi_zone_map_bronx.jpg')
print(image.shape)
print(image[700][1010])
# RGB color for green part [201 242 208]
#io.imshow(image)
#io.imsave('./test.jpg',image)

binary_img = np.zeros((4, 7), dtype=np.int16)
index = [[0, 2], [0, 5],
        [1, 0], [1, 1], [1, 2], [1, 4], [1, 5], [1, 6],
        [2, 2], [2, 5],
        [3, 1], [3, 2], [3, 4], [3, 6]]
for i in index:
    binary_img[i[0],i[1]] = np.int16(255)
print(binary_img)
io.imshow(binary_img)
plt.show()