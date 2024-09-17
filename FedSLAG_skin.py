import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch import optim
from unet.unet_model_feat import UNet
from torch.utils.data import Dataset, DataLoader

from IPython.display import clear_output
from utils.federated_tools import *
from utils.log_tools import *
from utils.SPB.tools import *
from configs.skin.FedSLAG import *
# %%

for client in skin_dataset:
    tmp = skin_dataset[client]
    x_, y_ = [DIR_DATA + f + '.jpg' for f in tmp], \
        [DIR_GT + f + '.png' for f in tmp]
    x_train, x_test, y_train, y_test = train_test_split(
        x_, y_, test_size=1 - TRAIN_RATIO, random_state=RS)

    split_dataset[client + '_train'] = Cancer(x_train, y_train, train=True, \
                                              IMAGE_SIZE=IMAGE_SIZE \
                                              , CROP_SIZE=CROP_SIZE)
    STATIC_WEIGHT[order] = len(x_train)
    order += 1

    split_dataset[client + '_test'] = Cancer(x_test, y_test, train=False, \
                                             IMAGE_SIZE=IMAGE_SIZE \
                                             , CROP_SIZE=CROP_SIZE)
    print(client)

VDIAS_LEN = STATIC_WEIGHT[2]
# %%
STATIC_WEIGHT = [item / sum(STATIC_WEIGHT) for item in STATIC_WEIGHT]
print(STATIC_WEIGHT)
WEIGHTS = STATIC_WEIGHT
WEIGHTS_DATA = copy.deepcopy(WEIGHTS)
# %%
device = torch.device('cuda:0')
LR, WD, TH = 1e-3, 1e-4, 0.9
best_avg_acc, best_epoch = 0.0, 0
# %%
training_clients, testing_clients = dict(), dict()
training_clients_pl = dict()

acc_train, acc_test, loss_train, loss_test = dict(), dict(), \
    dict(), dict()

nets, optimizers = dict(), dict()

nets['global'] = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=True).to(DEVICE)
nets_2['global'] = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=True).to(DEVICE)


print('Loaded the global model.')

for client in CLIENTS:

    training_loader[client] = DataLoader(split_dataset[client + '_train'], batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=8)
    training_loader_pl[client] = DataLoader(split_dataset[client + '_train'], batch_size=1,
                                            shuffle=True,
                                            num_workers=8)
    testing_loader[client] = DataLoader(split_dataset[client + '_test'], batch_size=1,
                                        shuffle=False, num_workers=1)
    acc_train[client], acc_test[client] = [], []
    loss_train[client], loss_test[client] = [], []
    acc_test_local[client], loss_test_local[client] = [], []

    nets[client] = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=True).to(DEVICE)
    nets_2[client] = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=True).to(DEVICE)
    nets_history[client] = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=True).to(DEVICE).eval()


    optimizers[client] = optim.AdamW(nets[client].parameters(), lr=LR, weight_decay=WD)
    optimizers_2[client] = optim.AdamW(nets_2[client].parameters(), lr=LR, weight_decay=WD)


print('Finished creating local models.')

print('Init Aggre Weights of each Client:', WEIGHTS_CL)

select_data_ratios = [0, 0, 1.0, 0]

print('==== Start training ====')
for epoch in range(EPOCHS):
    index.append(epoch)
    SELECT_DATA = [0, 0, VDIAS_LEN, 0]
    start_time = time.time()

    if epoch < 0.95 *EPOCHS:
        fed_broadcast(CLIENTS, nets, epoch, best_epoch, step2_flag= step2_flag, data_ps=select_data_ratios, super_types=CLIENTS_SUPERVISION, fed_name='global')  # clients[] list do not contain 'global'
        fed_broadcast(CLIENTS_2, nets_2, epoch, best_epoch, step2_flag= step2_flag, data_ps=select_data_ratios, super_types=CLIENTS_SUPERVISION, fed_name='global')  # todo 作者独有

    for order, client in enumerate(CLIENTS):
        supervision = 'labeled'
        if CLIENTS_SUPERVISION[order] == 'labeled':
            continue
        else:
            supervision = CLIENTS_SUPERVISION[order]


        im_store, y_store, pl1_store, pl2_store = [], [], [], []
        count_, _ac = select_pl(nets['global'], nets_2['global'], DEVICE,
                                training_loader_pl[client], im_store, y_store, pl1_store,
                                pl2_store, TH=TH, supervision=supervision)

        SELECT_DATA[order] = count_

        if len(im_store) >= 1:
            tmp_dataset = cancer_v2(im_store, y_store, pl1_store, pl2_store, super_type=supervision)
            training_loader[client] = DataLoader(tmp_dataset, batch_size=16,shuffle=True, num_workers=8)
    
    select_data_ratios.clear()
    for i, client in enumerate(CLIENTS):
        total_data = len(split_dataset[client + '_train'])
        selected_data = SELECT_DATA[i]
        if total_data == 0:
            select_data_ratio = 0
        else:
            select_data_ratio = selected_data / total_data
        select_data_ratios.append(select_data_ratio)

    for order, (client, supervision_t) in enumerate(zip(CLIENTS, CLIENTS_SUPERVISION)):
        print('Training Client:', client, ' ',supervision_t)

        if supervision_t == 'labeled':

            local_train_f(training_loader[client], nets[client], nets['global'], optimizers[client], DEVICE,
                          acc=acc_train[client], loss=loss_train[client], epoch=epoch,
                          supervision_type=supervision_t, Local_epoch=LOCAL_EPOCH, net_history = nets_history[client])
            local_train_f(training_loader[client], nets_2[client], nets_2['global'], optimizers_2[client],
                          DEVICE,
                          acc=None, loss=None, epoch=epoch,
                          supervision_type=supervision_t, Local_epoch=LOCAL_EPOCH)

        else:
            local_train_w(training_loader[client], nets[client], nets['global'], optimizers[client], DEVICE,
                          acc=acc_train[client], loss=loss_train[client], epoch=epoch,
                          supervision_type=supervision_t, FedMix_network=1, Local_epoch=LOCAL_EPOCH, net_history = nets_history[client])
            local_train_w(training_loader[client], nets_2[client], nets_2['global'], optimizers_2[client],
                          DEVICE,
                          acc=None, loss=None, epoch=epoch,
                          supervision_type=supervision_t, FedMix_network=2, Local_epoch=LOCAL_EPOCH, net_history = nets_history[client])

        score[order] = loss_train[client][-1] ** BETA_

    denominator = sum(score)
    score = [s / denominator for s in score]

    print('Chosen pseudo labels:', SELECT_DATA)

    denominator = sum(SELECT_DATA)
    WEIGHTS_CL = [s / denominator for s in SELECT_DATA]

    for order, _ in enumerate(WEIGHTS):
        WEIGHTS[order] = WEIGHTS_CL[order] + LAMBDA_ * score[order]

    denominator = sum(WEIGHTS)
    WEIGHTS = [w / denominator for w in WEIGHTS]
    print('Aggr Weights:', WEIGHTS)


    if epoch < 0.97 * EPOCHS:
        fed_aggr(CLIENTS, WEIGHTS, nets, fed_name='global')
        fed_aggr(CLIENTS_2, WEIGHTS, nets_2, fed_name='global')



    avg_acc = 0.0
    for client in CLIENTS:

        test(epoch, testing_loader[client], nets['global'], DEVICE, acc_test[client],
             loss_test[client])
        avg_acc += acc_test[client][-1]
    avg_acc = avg_acc / TOTAL_CLIENTS

    avg_acc_local = 0.0
    for client in CLIENTS:

        test_local(epoch, testing_loader[client], nets[client], DEVICE, acc_test_local[client],
             loss_test_local[client])
        avg_acc_local += acc_test_local[client][-1]
    avg_acc_local = avg_acc_local / TOTAL_CLIENTS

    if check_acc_test(acc_test, 0.50) and check_acc_test(acc_test_local, 0.50) and step2_flag == False:
        print("\nSTEP 2 >",epoch)
        step2_flag = True

    if avg_acc > best_avg_acc:
        best_avg_acc = avg_acc
        best_epoch = epoch
        if best_epoch > 0.7 * EPOCHS:
            save_model(SAVE_MODEL_PATH, best_epoch, nets, acc_train, isbest=True) # BEST MODEL


    clear_output(wait=True)
    print('Epoch:', epoch, '|', 'Global_acc:', avg_acc, '|','Local_acc:', avg_acc_local, '|' 'Best_global_acc:', best_avg_acc)
    print('Epoch:', epoch, '|', 'acc_train:', acc_train['ROSENDAHL'][-1], '|',acc_train['VIDIR_MODERN'][-1], '|', acc_train['VIENNA_DIAS'][-1], '|',
          acc_train['VIDIR_MOLEMAX'][-1])
    print('Epoch:', epoch, '|', 'loss_train:', loss_train['ROSENDAHL'][-1], '|',loss_train['VIDIR_MODERN'][-1], '|', loss_train['VIENNA_DIAS'][-1], '|',
          loss_train['VIDIR_MOLEMAX'][-1])
    print('Epoch:', epoch, '|', 'global_test:', acc_test['ROSENDAHL'][-1], '|', acc_test['VIDIR_MODERN'][-1], '|', acc_test['VIENNA_DIAS'][-1], '|',
          acc_test['VIDIR_MOLEMAX'][-1])
    print('Epoch:', epoch, '|', 'local_test:',acc_test_local['ROSENDAHL'][-1], '|', acc_test_local['VIDIR_MODERN'][-1], '|', acc_test_local['VIENNA_DIAS'][-1], '|',
          acc_test_local['VIDIR_MOLEMAX'][-1])
    print(f"Epoch {epoch} cost {time.time() - start_time :.2f} seconds")
    print("===================================================================")
    

    acc_avg, loss_avg = 0, 0
    save_data_to_file(CLIENTS, acc_test,loss_train, SAVE_LOG_PATH + '/' + LOG_FILE_NAME)



plt.figure(0)
plt.plot(index, acc_train['ROSENDAHL'], colors[0], label='ROSENDAHL train',linewidth = linewidth)
plt.plot(index, acc_train['VIDIR_MODERN'], colors[1], label='VIDIR_MODERN train',linewidth = linewidth)
plt.plot(index, acc_train['VIENNA_DIAS'], colors[2], label='VIENNA_DIAS train',linewidth = linewidth)
plt.plot(index, acc_train['VIDIR_MOLEMAX'], colors[3], label='VIDIR_MOLEMAX train',linewidth = linewidth)
plt.grid(True)
plt.legend()
plt.savefig(SAVE_LOG_PATH + '/' + 'train_acc_curve.png')
# plt.show()

plt.figure(1)
plt.plot(index, acc_test['ROSENDAHL'], colors[0], label='ROSENDAHL test',linewidth = linewidth)
plt.plot(index, acc_test['VIDIR_MODERN'], colors[1], label='VIDIR_MODERN test',linewidth = linewidth)
plt.plot(index, acc_test['VIENNA_DIAS'], colors[2], label='VIENNA_DIAS test',linewidth = linewidth)
plt.plot(index, acc_test['VIDIR_MOLEMAX'], colors[3], label='VIDIR_MOLEMAX test',linewidth = linewidth)
plt.plot(index, [(a+b+c+d) / 4 for a,b,c,d in zip(acc_test['VIDIR_MOLEMAX'], acc_test['ROSENDAHL'],acc_test['VIDIR_MODERN'],acc_test['VIENNA_DIAS'])],
         colors[4], label='global test',linewidth = 1)
plt.grid(True)
plt.legend()
plt.savefig(SAVE_LOG_PATH + '/' + 'test_acc_curve.png')
# plt.show()

# 绘制本地准确度曲线
plt.figure(2)
plt.plot(index, acc_test_local['ROSENDAHL'], colors[0], label='ROSENDAHL local test',linewidth = linewidth)
plt.plot(index, acc_test_local['VIDIR_MODERN'], colors[1], label='VIDIR_MODERN local test',linewidth = linewidth)
plt.plot(index, acc_test_local['VIENNA_DIAS'], colors[2], label='VIENNA_DIAS local test',linewidth = linewidth)
plt.plot(index, acc_test_local['VIDIR_MOLEMAX'], colors[3], label='UDIAT local test',linewidth = linewidth)
plt.plot(index, [(a+b+c+d) / 4 for a,b,c,d in zip(acc_test_local['VIDIR_MOLEMAX'], acc_test_local['ROSENDAHL'],acc_test_local['VIDIR_MODERN'],acc_test_local['VIENNA_DIAS'])],
          colors[4], label='global test',linewidth = 1)
plt.grid(True)
plt.legend()

plt.savefig(SAVE_LOG_PATH + '/' + 'acc_test_local_curve.png')
# plt.show()

print('Best_glob_acc and epoch:', best_avg_acc, best_epoch)

for client in CLIENTS:
    tmp = best_epoch
    best_epoch = best_epoch

    print(f"##{client}##")
    print("Shared epoch specific:", acc_test[client][best_epoch])
    print("Max client-specific:", np.max(acc_test[client]))
    best_epoch = tmp


print(time.strftime('Finish:%Y-%m-%d %H:%M:%S', time.localtime(time.time())))





