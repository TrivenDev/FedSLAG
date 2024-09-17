from torch import optim
from unet import UNet

####################################################
# for data preparation
####################################################

# 用于数据准备的库
from torch.utils.data import DataLoader
# 用于绘图的库

# 自定义的辅助函数
from utils.log_tools import *
####################################################
# for plotting
####################################################
from IPython.display import clear_output
############################
# Helper func
############################
from utils.federated_tools import *
# from configs.breast.FedAvg_config import *

MODEL_DIR = "backup/Train_Log/FedNEWGMM_bbbl_7735/FedNEWGMM_bbbl_2024-08-07_18-26-54/BUSIS/_best_2024-08-08.pth"
print("Testing Model:{}",MODEL_DIR)

# 设置一些全局常量和超参数
DATE = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
# LOG_FILE_NAME = ALGO_NAME + '_train_log.json'


DIR = 'dataset/breast'  # 数据集目录
CLIENTS = ['BMC', 'BUS', 'BUSIS', 'UDIAT'] # todo 客户端名称
CLIENTS_2 = [cl for cl in CLIENTS]  # 客户端名称 副本
# 不同客户端的监督方式
CLIENTS_SUPERVISION = ['labeled','labeled', 'labeled', 'labeled']
SUPERVISION_FLAG = get_supervision_flag(CLIENTS_SUPERVISION)
TOTAL_CLIENTS = len(CLIENTS)
# 分类任务路径：数据集目录+数据源+classification.
DIR_CLASSIFICATION = DIR + '/BUS/classification'

# 初始化实验参数
TRAIN_RATIO = 0.8
RS = 30448  # 随机种子
N_CHANNELS, N_CLASSES = 1, 1  #
IMAGE_SIZE = (256, 256)  # 图像尺寸
CROP_SIZE = (224, 224)  # 裁剪尺寸
CONSISTENT_PATH = 'breast_datasplit.npy'
DEVICE = torch.device('cuda:0')  # 使用 GPU0
FOLD_VERSION = 1
# 读取训练集和测试集的文件名单
consistent_path = np.load(CONSISTENT_PATH, allow_pickle=True).item()
breast_dataset = dict()
fivefoldacc=0

# load model
MODEL = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, \
                      bilinear=True).to(DEVICE)
# 加载模型参数
MODEL.load_state_dict(torch.load(MODEL_DIR, map_location=DEVICE))
MODEL.eval()

for FOLD_VERSION in range(5):
    print("=================FOLD VERSION:{}=================".format(FOLD_VERSION))
    idx_ = 0
    denom_ = 0
    for client, sup in zip(CLIENTS, CLIENTS_SUPERVISION): # [[A,label],[B,label]]
        dir_of_interest = consistent_path[client][FOLD_VERSION+1]

        x_test = dir_of_interest['x_test']
        y_test = dir_of_interest['y_test']
        idx_ += 1
        breast_dataset[client + '_test'] = Cancer(x_test, y_test, train=False, \
                                                  IMAGE_SIZE=IMAGE_SIZE \
                                                  , CROP_SIZE=CROP_SIZE)

    # storage file
    testing_clients = dict()
    acc_test, loss_test = dict(), dict()

    # dataloader
    for client in CLIENTS:
        testing_clients[client] = DataLoader(breast_dataset[client + '_test'], batch_size=1, \
                                             shuffle=False, num_workers=0)
        print("load test data:{},{}".format(client,len(testing_clients[client])))
        acc_test[client] = []
        loss_test[client] = []

    ################### test ##############################
    avg_acc = 0.0
    for client in CLIENTS:
        print("Test client:", client)
        test(1, testing_clients[client], MODEL, DEVICE, acc_test[client], \
             loss_test[client])
        avg_acc += acc_test[client][-1]

    avg_acc = avg_acc / TOTAL_CLIENTS
    ############################################################
    ########################################################

    # 清除输出并打印训练进度
    clear_output(wait=True)
    print('Avg_DSC:', avg_acc)
    print('Acc_test:', acc_test['BMC'][-1], '|', acc_test['BUS'][-1], '|', acc_test['BUSIS'][-1], '|',
          acc_test['UDIAT'][-1])
    # print('Epoch:', epoch, '|', 'local_test:', acc_test_local['BUS'][-1], '|', acc_test_local['BUSIS'][-1], '|',
    #       acc_test_local['UDIAT'][-1])
    fivefoldacc  += avg_acc

############### 打印测试精度和最佳轮次 ##############
# 宣布结束
print('Five Fold ACC:',fivefoldacc/5)
print(time.strftime('Finish:%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

# 测试acc
# plot_graphs(3, CLIENTS, index, acc_test, ' acc_test')


