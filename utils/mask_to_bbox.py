import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
def mask2bbox(mask_tensor, min_area=100, max_displacement=2, noisy=True):
    """
    将图像中的矩形区域填充为白色。

    参数:
    - mask_tensor: 一个Tensor，代表需要处理的图像掩膜。
    - min_area: 最小区域面积阈值，默认为100。面积小于该值的连通组件将被忽略。
    - max_displacement: 随机扰动的最大偏移量，默认为5。用于给填充的矩形区域添加随机的位移。
    - noisy: 是否在填充矩形时添加随机扰动，默认为True。

    返回值:
    - bbox_mask: 一个和输入mask_tensor大小相同的Tensor，其中填充了白色矩形的区域为1，其余为0。
    """

    # 阈值化图像，转换为二值图像
    binary_image = (mask_tensor > 0).float()

    # 连通组件标记，统计每个连通组件的属性
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image.squeeze().numpy().astype(np.uint8), connectivity=8)

    # 过滤出面积大于等于min_area的连通组件
    separate_regions = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            separate_regions.append(i)

    # 构建一个全零的和mask_tensor大小相同的tensor
    bbox_mask = torch.zeros_like(mask_tensor)

    # 遍历过滤后的连通组件，计算并填充每个组件的最小外接矩形
    for region in separate_regions:
        region_mask = (labels == region).astype('uint8')
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            # 计算轮廓的最小外接矩形
            x, y, w, h = cv2.boundingRect(contours[0])
            # 可选地，为矩形添加随机扰动
            if noisy:
                x += np.random.randint(-max_displacement, max_displacement + 1)
                y += np.random.randint(-max_displacement, max_displacement + 1)
                w += np.random.randint(-max_displacement, max_displacement + 1)
                h += np.random.randint(-max_displacement, max_displacement + 1)
            # 将矩形区域填充为白色
            bbox_mask[:, y:y + h, x:x + w] = 1


    return bbox_mask

def read_image_as_tensor(image_path):
    # 使用Pillow读取图像
    image = Image.open(image_path).convert('L')  # 'L'模式表示灰度图像
    
    # 应用转换并增加一个批次维度
    image_tensor = TF.to_tensor(image)
    
    return image_tensor

# 测试 mask2bbox 函数
def test_mask2bbox(image_path):
    mask_tensor = read_image_as_tensor(image_path)
    print(mask_tensor)

    bbox_mask = mask2bbox(mask_tensor, min_area=100, max_displacement=2, noisy=True)
    
    # 将Tensor转换为numpy数组以供显示 output_mask_tensor.numpy().astype(np.uint8) * 255
    bbox_mask_np = bbox_mask.numpy().astype(np.uint8) * 255
    # print(bbox_mask)
    
    # 显示结果
    plt.imshow(bbox_mask_np, cmap='gray')  # cmap='gray'是为了灰度图显示
    plt.show()

if __name__ == '__main__':
    print('mask to bbox:')
    # test_mask2bbox('path_to_your_image.png')
    # test_mask2bbox('dataset/breast/BUS/GT/benign (1).png')

    test_image_path = "dataset/breast/BUS/GT/benign (1).png"
    test_image_pil = Image.open(test_image_path).convert('L')
    test_image_tensor = TF.to_tensor(test_image_pil)


    # 调用给定的函数，获取输出结果
    output_mask_tensor = mask2bbox(test_image_tensor)

    # 将输出结果保存为 PNG 格式的图像文件
    output_mask_np = output_mask_tensor.numpy().astype(np.uint8)  *255 # 转换为 numpy 数组，并将像素值缩放到 [0, 255] 范围
    output_mask_image = Image.fromarray(output_mask_np.squeeze(), mode="L")  # 创建 PIL Image 对象
    output_mask_image.save("output_begin01.png")  # 保存为 PNG 格式的图像文件

