import json
import os.path
import random
import time

import matplotlib.pyplot as plt
import torch

colors = ['r', 'g', 'b', 'c', 'k', 'y', 'm', 'c']

def generate_data():
    acc = {
        'FedMix_glob': [random.uniform(0.3, 0.6) + random.uniform(0, 0.1) for _ in range(20)],
        'FedMix_BUS': [random.uniform(0.1, 0.5) + random.uniform(0, 0.1) for _ in range(20)],
        'FedMix_UDIAT': [random.uniform(0.2, 0.6) + random.uniform(0, 0.1) for _ in range(20)],
        'FedMix_BUSIS': [random.uniform(0.3, 0.7) + random.uniform(0, 0.1) for _ in range(20)]
    }
    loss = {
        'FedMix_glob': [1.0 - random.uniform(0, 0.1) - random.uniform(0.3, 0.6) for _ in range(20)],
        'FedMix_BUS': [1.0 - random.uniform(0, 0.1) - random.uniform(0.1, 0.5) for _ in range(20)],
        'FedMix_UDIAT': [1.0 - random.uniform(0, 0.1) - random.uniform(0.2, 0.6) for _ in range(20)],
        'FedMix_BUSIS': [1.0 - random.uniform(0, 0.1) - random.uniform(0.3, 0.7) for _ in range(20)]
    }
    return acc, loss


def save_data_to_file(CLIENTS, acc, loss, filename):
    data = {clientname:{'acc': acc[clientname], 'loss': loss[clientname]} for clientname in CLIENTS}
    with open(filename, 'w') as f:
        json.dump(data, f)


def load_data_from_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data['acc'], data['loss']


def plot_data(acc_dict, loss_dict, total_epochs, save_dir=None):
    epochs = range(1, total_epochs + 1)
    plt.figure(figsize=(10, 5))
    for client, acc in acc_dict.items():
        plt.plot(epochs, acc,  label=f'{client}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Total')
    plt.legend()
    plt.show()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'acc_plot.png'))
    else:
        plt.show()

    plt.figure(figsize=(10, 5))
    for client, loss in loss_dict.items():
        plt.plot(epochs, loss, label=f'{client}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Total')
    plt.legend()
    plt.show()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    else:
        plt.show()


def get_supervision_flag(CLIENTS_SUPERVISION):
    result_flag = ''
    for i in range(len(CLIENTS_SUPERVISION)):
        if CLIENTS_SUPERVISION[i] == 'labeled':
            result_flag += 'l'
        elif CLIENTS_SUPERVISION[i] == 'unlabeled':
            result_flag += 'u'
        elif CLIENTS_SUPERVISION[i] == 'bbox':
            result_flag += 'b'
        elif CLIENTS_SUPERVISION[i] == 'img':
            result_flag += 'i'
        elif CLIENTS_SUPERVISION[i] == 'point':
            result_flag += 'p'
        elif CLIENTS_SUPERVISION[i] == 'scribble':
            result_flag += 's'

    return result_flag



def make_print_to_file(path='./'):
    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    import sys
    import os
    import sys
    import datetime

    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.path= os.path.join(path, filename)
            self.log = open(self.path, "a", encoding='utf8',)
            print("save:", os.path.join(self.path, filename))

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    fileName = datetime.datetime.now().strftime('day'+'%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)

    #############################################################
    # 这里输出之后的所有的输出的print 内容即将写入日志
    #############################################################
    print(fileName.center(60,'*'))


def save_model(PTH, epoch, nets, acc_train, isbest=False):
    TIME =time.strftime("%Y-%m-%d",time.localtime(time.time()))
    for client, _ in acc_train.items():

        clientpath = PTH + '/' + client + '/'
        if not os.path.exists(clientpath):
            os.makedirs(clientpath)

        if isbest is True:
            savepath = (clientpath +
                    '_' + 'best' + '_' + TIME + '.pth')
        else:
            savepath = clientpath + "epoch" + str(epoch) + '_' + TIME + '.pth'

        torch.save(nets[client].state_dict(), savepath)


def check_acc_val(acc_test, acc_test_threshold):
    for client, acc_list in acc_test.items():
        latest_acc = acc_list[-1]
        if latest_acc <= acc_test_threshold:
            return False
    return True
# if __name__ == '__main__':
#     # Example usage
#     LOG_PATH = '../log/TestDir/'
#     TOTAL_EPOCHS = 20
#
#     if not os.path.exists(LOG_PATH):
#         os.makedirs(LOG_PATH)
#
#     acc, loss = generate_data()
#     print(acc)
#     print(loss)
#
#     save_data_to_file(acc, loss, LOG_PATH + 'logdata.json')
#
#     loaded_acc, loaded_loss = load_data_from_file(LOG_PATH + 'logdata.json')
#
#     plot_data(loaded_acc, loaded_loss, TOTAL_EPOCHS, LOG_PATH)
