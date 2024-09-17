import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from PIL import Image
def mask2point(mask):
    mask_np = np.array(mask, dtype=np.uint8)
    mask_np = mask_np.squeeze()
    _, mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)  # 二值化

    # 初始化一个全灰色的图像用于点监督
    points_image = np.full_like(mask_np, 127)
    # grey_image= np.full_like(mask_np, 127)

    # 查找分割对象的轮廓
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 定义一个4x4的正方形区域
    offsets = [(i, j) for i in range(8) for j in range(8)]

    if len(contours) > 0:
        for contour in contours:
            # 获取轮廓的矩，判断目标是白色还是黑色
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = 0, 0

            x, y, w, h = cv2.boundingRect(contour)

            # 选择四个不同的位置，对于黑色目标使用黑色点，白色目标使用白色点
            # 根据质心颜色决定点的颜色

            cx_ratio = 0.7
            cy_ratio = 0.3
            cx_boundary = int(cx_ratio * cx + (1 - cx_ratio) * (x + w/2))
            cy_boundary = int(cy_ratio * cy + (1 - cy_ratio) * (y + h/2))

            strip_offsets = [
                (cx, cy_boundary - h//4),           # 正上方
                (cx, cy_boundary + h//4),           # 正下方
                (cx_boundary - w//4, cy),           # 正左方
                (cx_boundary + w//4, cy),           # 正右方
            ]

            strip_offsets_black = [
                (cx, cy_boundary - h//2),           # 正上方
                (cx, cy_boundary + h//2),           # 正下方
                (cx_boundary - w//2, cy),           # 正左方
                (cx_boundary + w//2, cy),           # 正右方
            ]

            # 生成点
            for (ox, oy) in strip_offsets:
                for (dx, dy) in offsets:
                    px = ox + dx
                    py = oy + dy

                    # 确保点在图像范围内且不与已有目标冲突（实际上，由于是按轮廓处理，这一点可能已由轮廓检测自然保证）
                    if 0 <= px < points_image.shape[1] and 0 <= py < points_image.shape[0]:
                        points_image[py, px] = 255

            for (ox, oy) in strip_offsets_black:
                for (dx, dy) in offsets:
                    px = ox + dx
                    py = oy + dy

                    # 确保点在图像范围内且不与已有目标冲突（实际上，由于是按轮廓处理，这一点可能已由轮廓检测自然保证）
                    if 0 <= px < points_image.shape[1] and 0 <= py < points_image.shape[0]:
                        points_image[py, px] = 0
    else: # full black
        x,y=mask_np.shape[1]//2,mask_np.shape[0]//2
        # print('mask shape xy:',x,y)
        cx, cy = x-25, y-25
        cx_ratio = 0.7
        cy_ratio = 0.3
        cx_boundary = int(cx_ratio * cx + (1 - cx_ratio) * (x + 20 / 2))
        cy_boundary = int(cy_ratio * cy + (1 - cy_ratio) * (y + 20 / 2))
        strip_offsets_black = [
            (cx, cy_boundary - 20 // 2),  # 正上方
            (cx, cy_boundary + 20 // 2),  # 正下方
            (cx_boundary - 20 // 2, cy),  # 正左方
            (cx_boundary + 20 // 2, cy),  # 正右方
        ]
        for (ox, oy) in strip_offsets_black:
            for (dx, dy) in offsets:
                px = ox + dx
                py = oy + dy

                # 确保点在图像范围内且不与已有目标冲突（实际上，由于是按轮廓处理，这一点可能已由轮廓检测自然保证）
                if 0 <= px < points_image.shape[1] and 0 <= py < points_image.shape[0]:
                    points_image[py, px] = 0
                    


    res = np.expand_dims(points_image, axis=0)  # 扩展为1*H*W

    # 根据实际需要调整输出值的映射
    res[res == 127] = 2
    res[res == 255] = 1  # 目标 白色
    res[res == 0] = 0  #  背景类 黑色
    res[res == 2] = 255  #  灰色 背景


    return torch.from_numpy(res).float()




# def mask2point(mask):
#     mask_np = np.array(mask, dtype=np.uint)
#     # print(mask_np.shape)
#     mask_np = mask_np.squeeze()
#     _, mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY) # 二值化


#     # 查找分割对象的轮廓
#     contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # print('findcontours')
#     # 初始化一个空白图像用于点监督
#     points_image = np.zeros_like(mask_np)

#     # 定义8x8的正方形区域
#     offsets = [(i, j) for i in range(8) for j in range(8)]

#     for contour in contours:
#         # 计算质心和边界框
#         M = cv2.moments(contour)
#         if M['m00'] != 0:
#             cx = int(M['m10'] / M['m00'])
#             cy = int(M['m01'] / M['m00'])
#         else:
#             cx, cy = 0, 0
        
#         x, y, w, h = cv2.boundingRect(contour)
#         # 计算质心到边界的距离
#         cx_ratio = 0.7
#         cy_ratio = 0.3
#         cx_boundary = int(cx_ratio * cx + (1 - cx_ratio) * (x + w/2))
#         cy_boundary = int(cy_ratio * cy + (1 - cy_ratio) * (y + h/2))

#         # 在分割区域内选择四个不同的位置
#         strip_offsets = [
#             (cx, cy_boundary - h//4),           # 正上方
#             (cx, cy_boundary + h//4),           # 正下方
#             (cx_boundary - w//4, cy),           # 正左方
#             (cx_boundary + w//4, cy),           # 正右方
#         ]

#         # 生成三个6x4的小长条，45度倾斜
#         for (ox, oy) in strip_offsets:
#             for (dx, dy) in offsets:
#                 px = ox + dx
#                 py = oy + dy
                
#                 # 确保点在图像范围内
#                 if 0 <= px < points_image.shape[1] and 0 <= py < points_image.shape[0]:
#                     points_image[py, px] = 255

#     points_image =  np.expand_dims(points_image, axis=0) # 扩展为1*224*224
#     return torch.from_numpy(points_image).float()