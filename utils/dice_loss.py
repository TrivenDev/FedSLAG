import torch
from torch.autograd import Function


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.reshape(-1), target.reshape(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """计算批量数据的Dice系数（Dice coefficient）"""
    # 检查输入张量是否在GPU上，如果是，创建一个零张量并将其移到GPU上；如果不是，创建一个零张量。
    i=0
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    # 遍历输入张量和目标张量中的每个样本。
    # zip是把两个二维数组的对应位置的元素组合成元组
    for i, c in enumerate(zip(input, target)):
        # 调用DiceCoeff类的forward方法计算每个样本的Dice系数，并将结果加到累加器s上。
        s = s + DiceCoeff().forward(c[0], c[1])

    # 返回的是累加器s的平均值，即计算批量数据的平均Dice系数。
    return s / (i + 1)
