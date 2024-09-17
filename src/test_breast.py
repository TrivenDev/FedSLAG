from torch import optim
from unet import UNet
from torch.utils.data import DataLoader

from utils.log_tools import *
from IPython.display import clear_output
from utils.federated_tools import *
# from configs.breast.FedAvg_config import *

MODEL_DIR = "path/to/your/model.pth"
print("Testing Model:{}",MODEL_DIR)

DATE = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
# LOG_FILE_NAME = ALGO_NAME + '_train_log.json'

DIR = 'path/to/your/dataset'
CLIENTS = ['BMC', 'BUS', 'BUSIS', 'UDIAT']
CLIENTS_2 = [cl for cl in CLIENTS]
CLIENTS_SUPERVISION = [' ',' ', ' ', ' ']
# SUPERVISION_FLAG = get_supervision_flag(CLIENTS_SUPERVISION)
TOTAL_CLIENTS = len(CLIENTS)
DIR_CLASSIFICATION = DIR + '/BUS/classification'

TRAIN_RATIO = 0.8
RS = 30448
N_CHANNELS, N_CLASSES = 1, 1  #
IMAGE_SIZE = (256, 256)
CROP_SIZE = (224, 224)
split_file = 'path/to/your/data_split.npy'
DEVICE = torch.device('cuda:0')
FOLD_VERSION = 1
consistent_path = np.load(split_file, allow_pickle=True).item()
breast_dataset = dict()
five_fold_acc=0

MODEL = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=True).to(DEVICE)
MODEL.load_state_dict(torch.load(MODEL_DIR, map_location=DEVICE))
MODEL.eval()

for FOLD_VERSION in range(5):
    print("=================FOLD VERSION:{}=================".format(FOLD_VERSION))
    idx_ = 0
    denom_ = 0
    for client, sup in zip(CLIENTS, CLIENTS_SUPERVISION):
        dir_of_interest = consistent_path[client][FOLD_VERSION+1]

        x_test = dir_of_interest['x_test']
        y_test = dir_of_interest['y_test']
        idx_ += 1
        breast_dataset[client + '_test'] = Cancer(x_test, y_test, train=False, IMAGE_SIZE=IMAGE_SIZE,CROP_SIZE=CROP_SIZE)

    testing_clients = dict()
    acc_test, loss_test = dict(), dict()

    for client in CLIENTS:
        testing_clients[client] = DataLoader(breast_dataset[client + '_test'], batch_size=1,shuffle=False, num_workers=1)
        print("load test data:{},{}".format(client,len(testing_clients[client])))
        acc_test[client] = []
        loss_test[client] = []

    ################### test ##############################
    avg_acc = 0.0
    for client in CLIENTS:
        print("Test client:", client)
        val(1, testing_clients[client], MODEL, DEVICE, acc_test[client],loss_test[client])
        avg_acc += acc_test[client][-1]
    avg_acc = avg_acc / TOTAL_CLIENTS


    clear_output(wait=True)
    print('Avg_DSC:', avg_acc)
    print('Acc_test:', acc_test['BMC'][-1], '|', acc_test['BUS'][-1], '|', acc_test['BUSIS'][-1], '|',
          acc_test['UDIAT'][-1])
    five_fold_acc += avg_acc


print('Five Fold ACC:', five_fold_acc / 5)
print(time.strftime('Finish:%Y-%m-%d %H:%M:%S', time.localtime(time.time())))



