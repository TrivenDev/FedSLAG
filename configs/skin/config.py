import os
import time
import numpy as np
import pandas as pd
import torch.multiprocessing
from sklearn.model_selection import StratifiedShuffleSplit

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
from utils.log_tools import get_supervision_flag, make_print_to_file

ALGO_NAME = "FedSLAG"
DATE = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
LOG_FILE_NAME = ALGO_NAME + '_train_log.json'


PATH_META = 'path/to/your/metadata'
DIR = 'path/to/your/data'
DIR_DATA = DIR + 'derma/'
DIR_GT = DIR + 'GT/'

CLIENTS = ['ROSENDAHL', 'VIDIR_MODERN', 'VIENNA_DIAS', 'VIDIR_MOLEMAX']
CLIENTS_2 = [cl for cl in CLIENTS]

CLIENTS_SUPERVISION = ['labeled', 'labeled', 'labeled', 'bbox']


SUPERVISION_FLAG = get_supervision_flag(CLIENTS_SUPERVISION)
TOTAL_CLIENTS = len(CLIENTS)

TRAIN_RATIO = 0.8
RS = 30448
N_CHANNELS, N_CLASSES = 1, 1  #
bilinear = True  #
BATCH_SIZE, EPOCHS = 32, 300
LOCAL_EPOCH, NUM_WORKER = 1, 8
IMAGE_SIZE = (256, 256)
CROP_SIZE = (224, 224)
DEVICE = torch.device('cuda:0')
LR, WD = 1e-3, 1e-4
LAM, BETA, TH = 10, 1.5, 0.9
# FOLD_VERSION = 1
WEIGHTS_CL = [0.0, 0.0, 0.0, 0.0]  # 客户端的权重

print('Batch_size:',BATCH_SIZE,' EPOCH:', EPOCHS)


# consistent_path = np.load(CONSISTENT_PATH, allow_pickle=True).item()
# breast_dataset = dict()
idx_, denom_ = 0, 0
# 初始化模型、优化器和数据加载器
training_loader, testing_loader = dict(), dict()
training_loader_pl = dict()

acc_train, acc_test, loss_train, loss_test = dict(), dict(), dict(), dict()
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
# local_best_epoch= dict()
local_best_acc = dict()


LAMBDA_ = LAM
BETA_ = BETA
TH = TH

best_avg_acc = 0
best_epoch = 0
index = []

score = [0., 0., 0.,0.]  #
WEIGHTS = [0., 0., 0.,0.]  #


derma_ = pd.read_csv(PATH_META)
C1_rosendahl_idx = derma_['dataset'] == 'rosendahl'
C2_vidir_modern_idx = derma_['dataset'] == 'vidir_modern'
C3_vienna_dias_idx = derma_['dataset'] == 'vienna_dias'
C4_vidir_molemax_idx = derma_['dataset'] == 'vidir_molemax'

skin_dataset = dict()
skin_dataset['ROSENDAHL'] = derma_['image_id'][C1_rosendahl_idx]
skin_dataset['VIDIR_MODERN'] = derma_['image_id'][C2_vidir_modern_idx]
skin_dataset['VIENNA_DIAS'] = derma_['image_id'][C3_vienna_dias_idx]
skin_dataset['VIDIR_MOLEMAX'] = derma_['image_id'][C4_vidir_molemax_idx]

split_dataset = dict()
STATIC_WEIGHT = [0, 0, 0, 0]
order = 0

sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - TRAIN_RATIO, random_state=RS)

lesion_type_dict = {
    'nv': 'Melanocytic nevi',  # 0
    'bkl': 'Benign keratosis-like lesions ',
    'mel': 'dermatofibroma',
    'vasc': 'Vascular lesions',  # 3
    'bcc': 'Basal cell carcinoma',  # 4
    'akiec': 'Actinic keratoses',
    'df': 'Dermatofibroma'
}
lesion_type_dict_malignant = {
    'nv': 'ben',  # 0
    'bkl': 'ben',
    'df': 'ben',
    'vasc': 'ben',  # 3
    'bcc': 'mal',  # 4
    'akiec': 'mal',
    'mel': 'mal'
}