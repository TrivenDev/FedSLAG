import os
import time
import numpy as np
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch
from utils.log_tools import get_supervision_flag, make_print_to_file

ALGO_NAME = "FedSLAG"
DATE = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
LOG_FILE_NAME = ALGO_NAME + '_train_log.json'


DIR = 'path/to/your/data'
CLIENTS = ['BMC', 'BUS', 'BUSIS', 'UDIAT']
CLIENTS_2 = [cl for cl in CLIENTS]

CLIENTS_SUPERVISION = ['labeled', 'labeled', 'bbox','labeled']
SUPERVISION_FLAG = get_supervision_flag(CLIENTS_SUPERVISION)
TOTAL_CLIENTS = len(CLIENTS)
DIR_CLASSIFICATION = DIR + '/BUS/classification'

TRAIN_RATIO = 0.8
RS = 30448
N_CHANNELS, N_CLASSES = 1, 1
bilinear = True
BATCH_SIZE, EPOCHS = 16, 300
LOCAL_EPOCH, NUM_WORKER = 1, 8
IMAGE_SIZE = (256, 256)
CROP_SIZE = (224, 224)
CONSISTENT_PATH = 'path/to/your/data_split_file'
DEVICE = torch.device('cuda:0')
LR, WD = 1e-3, 1e-4
LAM, BETA, TH = 10, 1.5, 0.9
FOLD = 1
WEIGHTS_CL = [0.0, 0.0, 0.0, 0.0]


consistent_path = np.load(CONSISTENT_PATH, allow_pickle=True).item()
datasets = dict()
idx_, denom_ = 0, 0
training_loader, testing_loader = dict(), dict()
training_loader_pl = dict()
acc_train, acc_val, loss_train, loss_test = dict(), dict(), dict(), dict()
acc_val_local, loss_test_local = dict(), dict()
nets, optimizers = dict(), dict()
nets_2, optimizers_2 = dict(), dict()
nets_history = dict()


SAVE_MODEL_PATH = 'log/' + ALGO_NAME + '_' + SUPERVISION_FLAG + '_' + DATE
SAVE_LOG_PATH = SAVE_MODEL_PATH
os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
linewidth = 0.5
make_print_to_file(path=SAVE_LOG_PATH)

step2_flag = False

local_best_acc = dict()


LAMBDA_ = LAM
BETA_ = BETA
TH = TH

best_avg_acc = 0
best_epoch = 0
index = []

score = [0., 0., 0.,0.]
WEIGHTS = [0., 0., 0.,0.]


print('Batch_size:',BATCH_SIZE,' EPOCH:', EPOCHS)
