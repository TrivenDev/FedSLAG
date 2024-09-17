import glob
import math
import os
import random
import sys
import time

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize,dilation,square
import torch

sys.setrecursionlimit(1000000)
seed = 2024
np.random.seed(seed)
random.seed(seed)


def random_rotation(image, max_angle=15):
    angle = np.random.uniform(-max_angle, max_angle)
    img = Image.fromarray(image)
    img_rotate = img.rotate(angle)
    return np.array(img_rotate)


def translate_img(img, x_shift, y_shift):
    (height, width) = img.shape[:2]
    matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    trans_img = cv2.warpAffine(img, matrix, (width, height))
    return trans_img


def get_largest_two_component_2D(img, print_info=False, threshold=None):
    '''
    img: 二维数组
    return: 二维数组，最大两个连通域
    '''
    s = ndimage.generate_binary_structure(2, 2)
    labeled_array, numpatches = ndimage.label(img, s)
    sizes = ndimage.sum(img, labeled_array, range(1, numpatches + 1))
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    if print_info:
        print('component size', sizes_list)
    if len(sizes) == 1:
        out_img = [img]
    else:
        if threshold:
            max_size1 = sizes_list[-1]
            max_label1 = np.where(sizes == max_size1)[0] + 1
            if max_label1.shape[0] > 1:
                max_label1 = max_label1[0]
            component1 = labeled_array == max_label1
            out_img = [component1]
            for temp_size in sizes_list:
                if temp_size > threshold:
                    temp_lab = np.where(sizes == temp_size)[0] + 1
                    temp_cmp = labeled_array == temp_lab[0]
                    out_img.append(temp_cmp)
            return out_img
        else:
            max_size1 = sizes_list[-1]
            max_size2 = sizes_list[-2]
            max_label1 = np.where(sizes == max_size1)[0] + 1
            max_label2 = np.where(sizes == max_size2)[0] + 1
            if max_label1.shape[0] > 1:
                max_label1 = max_label1[0]
            if max_label2.shape[0] > 1:
                max_label2 = max_label2[0]
            component1 = labeled_array == max_label1
            component2 = labeled_array == max_label2
            if max_size2 * 10 > max_size1:
                out_img = [component1, component2]
            else:
                out_img = [component1]
    return out_img


class Cutting_branch:
    def __init__(self):
        self.lst_bifur_pt = 0
        self.branch_state = 0
        self.lst_branch_state = 0
        self.direction2delta = {0: [-1, -1], 1: [-1, 0], 2: [-1, 1], 3: [0, -1], 4: [0, 0], 5: [0, 1], 6: [1, -1], 7: [1, 0], 8: [1, 1]}

    def __find_start(self, lab):
        y, x = lab.shape
        idxes = np.asarray(np.nonzero(lab))
        for i in range(idxes.shape[1]):
            pt = tuple([idxes[0, i], idxes[1, i]])
            assert lab[pt] == 1
            directions = []
            for d in range(9):
                if d == 4:
                    continue
                if self.__detect_pt_bifur_state(lab, pt, d):
                    directions.append(d)
            if len(directions) == 1:
                start = pt
                self.start = start
                self.output[start] = 1
                return start
        start = tuple([idxes[0, 0], idxes[1, 0]])
        self.output[start] = 1
        self.start = start
        return start

    def __detect_pt_bifur_state(self, lab, pt, direction):
        d = direction
        y = pt[0] + self.direction2delta[d][0]
        x = pt[1] + self.direction2delta[d][1]
        if lab[y, x] > 0:
            return True
        else:
            return False

    def __detect_neighbor_bifur_state(self, lab, pt):
        directions = []
        for i in range(9):
            if i == 4:
                continue
            if self.output[tuple([pt[0] + self.direction2delta[i][0], pt[1] + self.direction2delta[i][1]])] > 0:
                continue
            if self.__detect_pt_bifur_state(lab, pt, i):
                directions.append(i)

        if len(directions) == 0:
            self.end = pt
            return False
        else:
            direction = random.sample(directions, 1)[0]
            next_pt = tuple([pt[0] + self.direction2delta[direction][0], pt[1] + self.direction2delta[direction][1]])
            if len(directions) > 1 and pt != self.start:
                self.lst_output = self.output * 1
                self.previous_bifurPts.append(pt)
            self.output[next_pt] = 1
            pt = next_pt
            self.__detect_neighbor_bifur_state(lab, pt)

    def __detect_loop_branch(self, end):
        for d in range(9):
            if d == 4:
                continue
            y = end[0] + self.direction2delta[d][0]
            x = end[1] + self.direction2delta[d][1]
            if (y, x) in self.previous_bifurPts:
                self.output = self.lst_output * 1
                return True

    def __call__(self, lab, seg_lab, iterations=1):
        self.previous_bifurPts = []
        self.output = np.zeros_like(lab)
        self.lst_output = np.zeros_like(lab)
        components = get_largest_two_component_2D(lab, threshold=15)
        if len(components) > 1:
            for c in components:
                start = self.__find_start(c)
                self.__detect_neighbor_bifur_state(c, start)
        else:
            c = components[0]
            start = self.__find_start(c)
            self.__detect_neighbor_bifur_state(c, start)
        self.__detect_loop_branch(self.end)
        struct = ndimage.generate_binary_structure(2, 2)
        output = ndimage.morphology.binary_dilation(
            self.output, structure=struct, iterations=iterations)
        shift_y = random.randint(-6, 6)
        shift_x = random.randint(-6, 6)
        if np.sum(seg_lab) > 1000:
            output = translate_img(output.astype(np.uint8), shift_x, shift_y)
            output = random_rotation(output)
        output = output * seg_lab
        return output


def scrible_2d(label, iteration=[4, 10]):
    '''
    产生2D涂鸦
    '''
    lab = label
    skeleton_map = np.zeros_like(lab, dtype=np.int32) # 骨架图，0表示背景，1表示前景
    for i in range(lab.shape[0]):    # 遍历每一层平面 只有1层
        if np.sum(lab[i]) == 0: 
            continue
        struct = ndimage.generate_binary_structure(2, 2) # 一个二值腐蚀结构元素 todo
        # 判断是否需要腐蚀，iter必须合法
        if np.sum(lab[i]) > 400 and iteration != 0 and iteration != [0] and iteration is not None:
            # 腐蚀迭代，整数次
            iter_num = math.ceil(iteration[0] + random.random() * (12 - iteration[0]))
            # print('iternum:',iter_num)
            # 对当前层做腐蚀
            slic = ndimage.binary_erosion(lab[i], structure=struct, iterations=iter_num)
            
        else:
            slic = lab[i]
        # 骨架化处理
        sk_slice = skeletonize(slic,method='lee')
        sk_slice = dilation(sk_slice, square(2))
        sk_slice = np.asarray((sk_slice == 255), dtype=np.int8)
        # sk_slice = np.asarray((sk_slice == 255), dtype=np.int32)
        # 骨架图
        skeleton_map[i] = sk_slice
    return skeleton_map

def scribble4class(label, class_id, class_num, iteration=[4, 10], cut_branch=True):
    '''
    生成特定类别的涂鸦，classid=0,1
    '''
    label = (label == class_id)
    sk_map = scrible_2d(label, iteration=iteration) # 白色表示涂鸦
    if cut_branch and class_id != 0: # 不是背景类
        cut = Cutting_branch()
        for i in range(sk_map.shape[0]):
            lab = sk_map[i]
            if lab.sum() < 1:
                continue
            sk_map[i] = cut(lab, seg_lab=label[i])
    if class_id == 0:
        class_id = class_num
    return sk_map * class_id


def generate_scribble(label, iterations, cut_branch=True):
    # label是mask图
    class_num = 2
    # print('cls num:',class_num)
    output = np.zeros_like(label, dtype=np.uint8)


    for i in range(class_num):
        it = iterations[i]if isinstance(iterations, list) else iterations  # 0,2 
        scribble = scribble4class(
            label, i, class_num, it, cut_branch=cut_branch)
        output += scribble.astype(np.uint8) #叠加？

    return output

# mask2scribble函数
def mask2scribble(label, iterations=[4, 10], cut_branch=True):
    class_num = 2  # 假设类别数为2，根据实际情况调整
    iter_up = 2 

    if not isinstance(label, np.ndarray):
        label_np = np.array(label)

    output = np.zeros_like(label_np, dtype=np.uint8)

    label_np[label_np==0] = 0
    label_np[label_np==255] = 1
    if label_np.ndim == 2: # 黑白？
        label_np = np.expand_dims(label_np, axis=0)
    
    # 遍历每个类别，生成scribble
    output = generate_scribble(label_np, iterations=tuple([1, iter_up-1]),cut_branch=False)
        # output += scribble.astype(np.uint8)
    
    # 根据实际需要调整输出值的映射
    output[output == 0] = 255  # 灰色 背景
    output[output == 1] = 1  # 目标 白色
    output[output == class_num] = 0  # 背景 黑色

    # output = np.squeeze(output, axis=0)
    output_torch = torch.from_numpy(output).float()
    
    return output_torch



# if __name__ == "__main__":
#     output_folder = "../data/breast/scribble/UDIAT"
#     os.makedirs(output_folder, exist_ok=True)

#     for image_path in sorted(glob.glob("../data/breast/UDIAT/*.png")):
#         print(f"Processing {image_path}")
#         num_classes = 2
#         iter_up =2

#         # 读取图像
#         mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#         # print(np.count_nonzero(mask),np.count_nonzero(mask==0)) 0是黑色 255是白色
#         mask[mask==0] = 0
#         mask[mask==255] = 1

#         if mask.ndim == 2: # 黑白？
#             mask = np.expand_dims(mask, axis=0)

#         # 涂鸦
#         scribble = generate_scribble(mask, iterations=tuple([1, iter_up-1]),cut_branch=False)


#         scribble[scribble == 0] = 127 # 灰色背景
#         scribble[scribble == 1] = 255 # 目标 白色
#         scribble[scribble == num_classes] = 0  # 背景 黑色

#         print('scribble shape:',scribble.shape)
#         scribble = np.squeeze(scribble, axis=0)  # Remove the extra dimension

#         # save the scribble image
#         output_path = os.path.join(output_folder, os.path.basename(image_path))
#         cv2.imwrite(output_path, scribble) 
#         print(f"Saved scribble to {output_path}")

#     print("Scribble generation completed.", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


