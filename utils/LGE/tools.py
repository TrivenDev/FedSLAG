import cv2
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
# from utils.LGE.utils import *



def build_cur_cls_label(mask, nclass):
    """some point annotations are cropped out, thus the prototypes are partial"""
    b = mask.size()[0] # batchsize
    mask_one_hot = one_hot(mask, nclass) # 输出一个新的张量，其中每个像素的类别表示为一个长度为 nclass 的向量，对应的类别位置为1，其余位置为0。 --  [1] or [0]
    cur_cls_label = mask_one_hot.view(b, nclass, -1).max(-1)[0] # 根据(b, nclass, -1)形态的独热标签，整理每个图有哪些类别出现过 --
    return cur_cls_label.view(b, nclass, 1, 1) # b个图片，每个图片用长度为nclass*1*1的列表，每个位置不是1就是0


def clean_mask(mask, cls_label, softmax=True):
    # clean_mask()函数的作用是对输入的mask进行清理，
    # 方法是将不在cls_label中的类别对应的mask全部去除，
    # 只保留在cls_label中出现过的类别对应的mask
    if softmax:
        mask = F.softmax(mask, dim=1)
    n, c = cls_label.size()
    """Remove any masks of labels that are not present"""
    return mask * cls_label.view(n, c, 1, 1)


def get_cls_loss(predict, cls_label, mask):
    """
    计算像素类别损失。

    参数:
    - predict: (b, k, h, w) 形状的张量，表示模型预测的概率。这里，b是批量大小，k是类别数，
               h和w分别是预测图的高度和宽度。
    - cls_label: (b, k) 形状的张量，表示每个批次中每个类别的存在性。如果某个类别在批次的任意图像中出现，
                 则相应的值为1，否则为0。
    - mask: (b, h, w) 形状的张量，含有真实的类别标签，用于计算损失。255代表该像素点应被忽略。

    过程:
    1. 首先，使用softmax函数对predict张量进行归一化处理，确保预测值表示概率。
    2. 将predict和mask张量调整形状，以便于计算损失。
    3. 计算损失，如果某个类别k在特定批次的图像中不存在（即cls_label[b, k]为0），
       那么该批次的所有像素点都不应该被分配给类别k。损失计算反映了这一点。
    4. 忽略mask中值为255的像素点，计算最终的平均损失。

    返回:
    - 损失值：一个标量，表示当前批次的平均类别损失。
    """

    b, k, h, w = predict.size()  # 提取张量维度信息

    predict = torch.softmax(predict, dim=1).view(b, k, -1)  # 对预测应用softmax并调整形状为三维
    mask = mask.view(b, -1)  # 调整mask的形状以匹配predict

    # print(b,k,h,w) # 2 1 224 224 debug
    # print(cls_label, cls_label.shape) # 2*1 []
    # print(f"predict shape: {predict.shape}") # 2*1*50176
    # print(cls_label)

    # 计算类别损失
    loss = - (1 - cls_label.view(b, k, 1)) * torch.log(1 - predict + 1e-6)
    loss = torch.sum(loss, dim=1)  # 按类别维度求和
    loss = loss[mask != 127].mean()  # 忽略mask中为255的像素点，并计算平均损失
    return loss

# 为一个批次b个图像标签的One-hot编码
def one_hot(label, nclass):
    # print(label.shape)
    b, k, h, w = label.size()
    label_cp = label.clone()

    label_cp[label > nclass] = nclass
    label_cp = label_cp.view(b, 1, h*w)

    mask = torch.zeros(b, nclass+1, h*w).to(label.device)
    mask = mask.scatter_(1, label_cp.long(), 1).view(b, nclass+1, h, w).float()
    return mask[:, :-1, :, :]
    '''
    类别0:
    [[1, 0],
    [0, 0]]

    类别1:
    [[0, 1],
    [0, 0]]

    类别2:
    [[0, 0],
    [1, 1]]
    '''


# 2D 标签的One-hot编码
def one_hot_2d(label, nclass):
    # print(label.shape)
    # 获取标签大小
    _ ,h, w = label.size()    

    # 对标签进行克隆以防止修改原始标签
    label_cp = label.clone()

    # 对于标签值大于类别数的部分赋值为nclass也就是背景类，这一步主要处理可能存在的异常标签值 如255
    label_cp[label > nclass] = nclass

    # 将标签展平成一行
    label_cp = label_cp.view(1, h*w) 

    # 在相应设备上创建全零的掩码张量，这个tensor有21+1行，h*w列 todo 为什么+1
    mask = torch.zeros(nclass+1, h*w).to(label.device)

    # 对每个标签进行one-hot编码，并通过scatter_()函数进行填充
    # 这里的scatter_函数会把标签位置的0改为1，表示该位置的类别
    # 最后再变回原始的二维形状 (nclass+1, h, w)，此时，每一个通道上的2D图分别表示某类别i的所属像素有哪些。值为1或者0.
    mask = mask.scatter_(0, label_cp.long(), 1).view(nclass+1, h, w).float()

    # 返回除最后一行（所有标签都为nclass的行）外的所有行，得到正常的one-hot编码
    return mask[:-1, :, :]


def cal_protypes(feat, mask, nclass):
    """
    计算每个类别的原型向量，并计算相应的损失。

    参数:
        feat (Tensor): 特征图，形状为 (batch_size, channels, height, width)
        mask (Tensor): y_weak掩码，形状为 (batch_size, height, width)
        nclass (int): 类别数目

    返回:
        prototypes (Tensor): 每个类别的原型向量，形状为 (batch_size, nclass, channels)
        loss (Tensor): 计算得到的损失
    """
    # 调整特征图的尺寸，与掩码的尺寸保持一致 双线性插值（bilinear）
    feat = F.interpolate(feat, size=mask.size()[-2:], mode='bilinear')

    # 获取特征图的维度 此时h和w应该和mask一样
    b, c, h, w = feat.size()

    # 初始化存储原型向量的张量 (batch_size, nclass, channels),channel是特征图的通道数，灰度图是1
    prototypes = torch.zeros((b, nclass, c), dtype=feat.dtype, device=feat.device)

    # 遍历批次的每个样本
    for i in range(b):
        # 当前掩码（单个图片nc*w*h）
        cur_mask = mask[i]
        cur_mask_onehot = one_hot_2d(cur_mask, nclass)  # 将掩码转换为one-hot表示nc*h*w 0黑 1白 255灰？？
        # 当前特征图
        cur_feat = feat[i]
        # 初始化当前批次的原型向量 1*C
        cur_prototype = torch.zeros((nclass, c), dtype=feat.dtype, device=feat.device)

        # 获取当前掩码中存在的类别set，没有重复元素
        cur_set = list(torch.unique(cur_mask))   # 0，1，255
        if 0 in cur_set: # todo 原来是nclass
            cur_set.remove(0)  # 移除黑色类别0
        if 255 in cur_set:
            cur_set.remove(255)


        # 遍历每个类别号
        for cls in cur_set:  # [1]
            m = cur_mask_onehot[0].view(1, h, w)  # nc*h*w取单个cls的部分，置为1*h*w 原来是cls.long() todo 避免下标有问题，直接写0
            sum = m.sum()  # 计算当前类别掩码的总和（有多少个像素点）
            
            # 扩展当前类别掩码，以匹配特征图的维度
            m = m.expand(c, h, w).view(c, -1)
            # 根据掩码计算该类别的特征向量并归一化
            cls_feat = (cur_feat.view(c, -1)[m == 1]).view(c, -1).sum(-1) / (sum + 1e-6)

            # 将计算得到的特征向量赋值给对应的原型向量 (对应类别的那行)
            cur_prototype[0, :] = cls_feat  # 原来是cls.long() todo 避免下标有问题，直接写0

        # 过完一个图片，当前批次的原型向量
        prototypes[i] += cur_prototype

    # 生成当前批次的类别标签
    cur_cls_label = build_cur_cls_label(mask, nclass).view(b, nclass, 1)
    # 计算类别的平均向量
    mean_vecs = (prototypes.sum(0) * cur_cls_label.sum(0)) / (cur_cls_label.sum(0) + 1e-6)

    # 计算原型损失
    loss = proto_loss(prototypes, mean_vecs, cur_cls_label)

    # 打印原型向量的形状
    # print('protype:',prototypes.shape, b, nclass, c)

    # 返回计算得到的原型向量和损失
    return prototypes.view(b, nclass, c), loss


def proto_loss(prototypes, vecs, cur_cls_label):
    b, nclass, c = prototypes.size()

    # abs = torch.abs(prototypes - vecs).mean(2)
    # positive = torch.exp(-(abs * abs))
    # positive = (positive*cur_cls_label.view(b, nclass)).sum()/(cur_cls_label.sum()+1e-6)
    # positive_loss = 1 - positive

    vecs = vecs.view(nclass, c)
    total_cls_label = (cur_cls_label.sum(0) > 0).long()
    negative = torch.zeros(1,
                           dtype=prototypes.dtype,
                           device=prototypes.device)

    num = 0
    for i in range(nclass):
        if total_cls_label[i] == 1:
            for j in range(i+1, nclass):
                if total_cls_label[j] == 1:
                    if i != j:
                        num += 1
                        x, y = vecs[i].view(1, c), vecs[j].view(1, c)
                        abs = torch.abs(x - y).mean(1)
                        negative += torch.exp(-(abs * abs))
                        # print(negative)

    negative = negative/(num+1e-6)
    negative_loss = negative

    return negative_loss


def GMM(feat, vecs, pred, true_mask, cls_label):
    b, k, oh, ow = pred.size()

    preserve = (true_mask < 255).long().view(b, 1, oh, ow)
    preserve = F.interpolate(preserve.float(), size=feat.size()[-2:], mode='bilinear')
    pred = F.interpolate(pred, size=feat.size()[-2:], mode='bilinear')
    _, _, h, w = pred.size()
    # print("pred shape:",pred.shape) # ([16, 1, 14, 14])

    # print('1- vec & feat',vecs.shape, feat.shape) # torch.Size([16, 2, 512]) torch.Size([16, 512, 14, 14])

    vecs = vecs.view(b, k, -1, 1, 1) # 这里pred的预测类别数量居然是1，人家是21
    feat = feat.view(b, 1, -1, h, w)

    # print('vec & feat',vecs.shape, feat.shape)
    #  # vec & feat torch.Size([16, 1, 1024, 1, 1]) torch.Size([16, 1, 512, 14, 14])
    """ 255 caused by cropping, using preserve mask """
    abs = torch.abs(feat - vecs).mean(2)
    abs = abs * cls_label.view(b, k, 1, 1) * preserve.view(b, 1, h, w)
    abs = abs.view(b, k, h*w)

    # """ calculate std """
    # pred = pred * preserve
    # num = pred.view(b, k, -1).sum(-1)
    # std = ((pred.view(b, k, -1)*(abs ** 2)).sum(-1)/(num + 1e-6)) ** 0.5
    # std = std.view(b, k, 1, 1).detach()

    # std = ((abs ** 2).sum(-1)/(preserve.view(b, 1, -1).sum(-1)) + 1e-6) ** 0.5
    # std = std.view(b, k, 1, 1).detach()

    abs = abs.view(b, k, h, w)
    res = torch.exp(-(abs * abs))
    # res = torch.exp(-(abs*abs)/(2*std*std + 1e-6))
    res = F.interpolate(res, size=(oh, ow), mode='bilinear')
    res = res * cls_label.view(b, k, 1, 1)

    return res


def loss_calc(preds, label, ignore_index, reduction='mean', multi=False, class_weight=False,
              ohem=False):
    """
    为语义分割计算交叉熵损失的函数。

    参数:
    preds (Tensor): 语义分割模型输出的预测。
    label (Tensor): 分割的真实标签。
    ignore_index (int): 应在损失计算中忽略的标签值。
    reduction (str): 指定应用于输出的缩减：'none' | 'mean' | 'sum'。
    multi (bool): 是否包含辅助输出，并且应在损失中考虑。
    class_weight (bool): 是否根据类频率使用加权损失。
    ohem (bool): 是否使用在线难例挖掘（OHEM）来计算损失。

    返回:
    Tensor: 计算出的损失值。
    """
    # 创建标签张量的副本，以便在不影响原始数据的情况下进行修改。
    label_cp = label.clone()
    # 将ignore_index位置的值设置为255，以便稍后告诉损失函数忽略这些位置。
    label_cp[label == ignore_index] = 255

    # 根据参数决定使用哪种交叉熵损失实例。
    if ohem:
        # 如果使用在线难例挖掘，初始化OHEM交叉熵损失。
        ce = OhemCrossEntropy(use_weight=True)
    else:
        # 检查是否应该将类权重应用到损失函数。
        if class_weight:
            # 如果是，为每个类定义权重（针对特定的数据集）。
            weight = torch.FloatTensor(
                [0.3, 0.5, 0.4762, 1.4286, 1.1111, 0.4762, 0.8333, 0.5, 0.5, 0.8333, 0.5263, 0.5882,
                 1.4286, 0.5, 3.3333, 5.0, 10.0, 2.5, 0.8333]).cuda()
            # 使用所提供的类权重初始化加权交叉熵损失。
            ce = torch.nn.CrossEntropyLoss(
                ignore_index=255, reduction=reduction, weight=weight)
        else:
            # 如果不使用类权重，直接初始化标准交叉熵损失。
            ce = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction) # pascal

    # 如果包含辅助预测（多输出），计算组合损失。
    if multi: # pascal - false
        aux_pred, pred = preds
        # 分别为辅助和主要预测计算损失，给它们分配40%和60%的权重。
        loss = ce(aux_pred, label_cp.long()) * 0.4 + ce(pred, label_cp.long()) * 0.6
    else:
        # 如果只有一个预测tensor，直接计算损失。
        loss = ce(preds, label_cp.long())

    # 返回计算出的损失。
    return loss


def cal_gmm_loss(pred, res, cls_label, true_mask):
    '''
    res:G from gmm
    prd:P from deeplab
    true_mask:y from p/s/full
    '''
    n, k, h, w = pred.size() # k 是 类别数量

    #
    loss1 = - res * torch.log(pred + 1e-6) - (1 - res) * torch.log(1 - pred + 1e-6)  # BCELOSS
    loss1 = loss1 / 2
    loss1 = (loss1*cls_label).sum(1) / (cls_label.sum(1)+1e-6) # 除以数量

    # print("shape:",loss1.shape, true_mask.shape)
    # true_mask_squeezed = true_mask.squeeze(0).squeeze(255)  # 去除1号维度 背景？？？？
    # print('cal gmm loss shape: ',loss1.shape, true_mask.shape) # cal gmm loss shape:  torch.Size([3, 224, 224]) torch.Size([3, 1, 224, 224])
    loss1 = loss1.unsqueeze(1)
    loss1 = loss1[true_mask != 255].mean()  # truemask

    true_mask_one_hot = one_hot(true_mask, k)
    loss2 = - true_mask_one_hot * torch.log(res + 1e-6) \
            - (1 - true_mask_one_hot) * torch.log(1 - res + 1e-6)
    loss2 = loss2/2
    try:
        loss2 = (loss2 * cls_label).sum(1) / (cls_label.sum(1) + 1e-6)
        loss2 = loss2.unsqueeze(1)
        loss2 = loss2[true_mask <= k].mean()  # <k
    except:
        print('cal gmm loss2 error:')
        print('true_mask:', true_mask.shape)
        print('pred:', pred.shape)
        print('res:', res.shape)
        print('cls_label:', cls_label.shape)
        print('tmoh:',true_mask_one_hot.shape)
        print('loss2:',loss2.shape)

    return (loss1+loss2) #加了个mean


class OhemCrossEntropy(nn.Module):
    """
    Ohem Cross Entropy Tensor Version
    """

    def __init__(
        self, ignore_index=255, thresh=0.7, min_kept=1e6, use_weight=False, reduce=False
    ):
        super(OhemCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            # weight = torch.FloatTensor(
            #     [
            #         0.8373,
            #         0.918,
            #         0.866,
            #         1.0345,
            #         1.0166,
            #         0.9969,
            #         0.9754,
            #         1.0489,
            #         0.8786,
            #         1.0023,
            #         0.9539,
            #         0.9843,
            #         1.1116,
            #         0.9037,
            #         1.0865,
            #         1.0955,
            #         1.0865,
            #         1.1529,
            #         1.0507,
            #     ]
            # ).cuda()
            # weight = torch.FloatTensor(
            #    [0.4762, 0.5, 0.4762, 1.4286, 1.1111, 0.4762, 0.8333, 0.5, 0.5, 0.8333, 0.5263, 0.5882,
            #    1.4286, 0.5, 3.3333,5.0, 10.0, 2.5, 0.8333]).to(label.device)
            weight = torch.FloatTensor(
                [0.3, 0.5, 0.4762, 1.4286, 1.1111, 0.4762, 0.8333, 0.5, 0.5, 0.8333, 0.5263, 0.5882,
                 1.4286, 0.5, 3.3333, 5.0, 10.0, 2.5, 0.8333]).cuda()
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="mean", weight=weight, ignore_index=ignore_index
            )
        elif reduce:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="none", ignore_index=ignore_index
            )
        else:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="mean", ignore_index=ignore_index
            )

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            pass
            # print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


if __name__ == '__main__':
    proto = torch.rand(8, 2, 256)
    vecs = torch.rand(1, 2, 256)
    cls_label = torch.rand(8, 2, 1)
    prl = proto_loss(proto, vecs, cls_label)

    print(proto)
    print(vecs)
    print(cls_label)
    print(prl)
    pass
