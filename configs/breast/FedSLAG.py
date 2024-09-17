import os
import time
import numpy as np
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch
from utils.log_tools import get_supervision_flag, make_print_to_file

ALGO_NAME = "FedNEW"
# 设置一些全局常量和超参数
DATE = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
LOG_FILE_NAME = ALGO_NAME + '_train_log.json'


DIR = 'dataset/breast'  # 数据集目录
CLIENTS = ['BMC', 'BUS', 'BUSIS', 'UDIAT']  # todo 客户端名称
CLIENTS_2 = [cl for cl in CLIENTS]  # 客户端名称 副本
# 不同客户端的监督方式
CLIENTS_SUPERVISION = ['scribble', 'scribble', 'bbox','labeled']
# CLIENTS_SUPERVISION = ['point', 'bbox', 'bbox','labeled']
SUPERVISION_FLAG = get_supervision_flag(CLIENTS_SUPERVISION)
TOTAL_CLIENTS = len(CLIENTS)
# 分类任务路径：数据集目录+数据源+classification.
DIR_CLASSIFICATION = DIR + '/BUS/classification'

# 初始化实验参数
TRAIN_RATIO = 0.8
RS = 30448  # 随机种子30448
N_CHANNELS, N_CLASSES = 1, 1  #
bilinear = True  #
BATCH_SIZE, EPOCHS = 16, 300
LOCAL_EPOCH, NUM_WORKER = 1, 8
IMAGE_SIZE = (256, 256)  # 图像尺寸
CROP_SIZE = (224, 224)  # 裁剪尺寸
CONSISTENT_PATH = 'breast_datasplit.npy'  # 如何生成
DEVICE = torch.device('cuda:0')  # 使用 GPU0
LR, WD = 1e-3, 1e-4  # 学习率和权重衰减
LAM, BETA, TH = 10, 1.5, 0.9  # hyper-parameters for adaptive aggregation
FOLD_VERSION = 1
WEIGHTS_CL = [0.0, 0.0, 0.0, 0.0]  # 客户端的权重


# 读取训练集和测试集的文件名单
consistent_path = np.load(CONSISTENT_PATH, allow_pickle=True).item()
datasets = dict()
idx_, denom_ = 0, 0
# 初始化模型、优化器和数据加载器
training_loader, testing_loader = dict(), dict()
training_loader_pl = dict()  # 训练数据加载器 {客户端名：Dataloader} 

acc_train, acc_test, loss_train, loss_test = dict(), dict(), dict(), dict()  # acc和loss表，每个客户端 每个epoch一个
acc_test_local, loss_test_local = dict(), dict()
nets, optimizers = dict(), dict()  # 存储各个神经网络对象和优化器对象
nets_2, optimizers_2 = dict(), dict()
nets_history = dict()

# 保存模型和日志的路径
SAVE_MODEL_PATH = 'log/' + ALGO_NAME + '_' + SUPERVISION_FLAG + '_' + DATE
SAVE_LOG_PATH = 'log/' + ALGO_NAME + '_' + SUPERVISION_FLAG + '_' + DATE
os.makedirs(SAVE_MODEL_PATH, exist_ok=True)  # 创建实验结果目录
linewidth = 0.5
make_print_to_file(path=SAVE_LOG_PATH)

# similarity_matrix = np.ones((len(CLIENTS), len(CLIENTS)))
step2_flag = False
# local_best_epoch= dict()
local_best_acc = dict()


# FedMix 相关的超参数和权重
LAMBDA_ = LAM
BETA_ = BETA
TH = TH

best_avg_acc = 0# 最好的acc及其epoch
best_epoch = 0
index = []  # todo 绘图的横坐标

score = [0., 0., 0.,0.]  #
WEIGHTS = [0., 0., 0.,0.]  #


print('Batch_size:',BATCH_SIZE,' EPOCH:', EPOCHS)
