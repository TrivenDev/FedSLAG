from torch import optim
from unet.unet_model_ import UNet
from torch.utils.data import DataLoader
from IPython.display import clear_output

from utils.federated_tools import *
from utils.log_tools import *
from utils.SPB.tools import *
from configs.breast.config import *

for client, sup in zip(CLIENTS, CLIENTS_SUPERVISION):
    data_split_dict = consistent_path[client][FOLD]
    x_train,x_test  = data_split_dict['x_train'],data_split_dict['x_test']
    y_train,y_test= data_split_dict['y_train'],data_split_dict['y_test']

    if sup == 'unlabeled':
        DATA_TYPE = ['original', 'GT']
        for _, _, files in os.walk(DIR + '/' + 'BUS' + '/classification/GT'):
            selected = [f for f in files if f[:6] == 'normal']
            for datatype in DATA_TYPE:
                tmp = [DIR + '/' + 'BUS' + '/classification/' + datatype + '/' + f for f in selected]
                if datatype == 'GT':
                    y_train += tmp
                else:
                    x_train += tmp

    print(client,' Train:',len(x_train),' / Test:',len(x_test))
    WEIGHTS_CL[idx_] = len(x_train)

    denom_ += len(x_train)
    idx_ += 1  # add

    datasets[client + '_train'] = Cancer(x_train, y_train, train=True,
                                         IMAGE_SIZE=IMAGE_SIZE, CROP_SIZE=CROP_SIZE)
    datasets[client + '_test'] = Cancer(x_test, y_test, train=False,
                                        IMAGE_SIZE=IMAGE_SIZE, CROP_SIZE=CROP_SIZE)

for idx_ in range(len(WEIGHTS_CL)):
    WEIGHTS_CL[idx_] = WEIGHTS_CL[idx_] / denom_

nets['global'] = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=True).to(DEVICE)
nets_2['global'] = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=True).to(DEVICE)
print('Loaded the global model.')

for client in CLIENTS:

    training_loader[client] = DataLoader(datasets[client + '_train'], batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=8)
    training_loader_pl[client] = DataLoader(datasets[client + '_train'], batch_size=1,
                                            shuffle=True,
                                            num_workers=8)
    testing_loader[client] = DataLoader(datasets[client + '_test'], batch_size=1,
                                        shuffle=False, num_workers=1)


    acc_train[client], acc_val[client] = [], []
    loss_train[client], loss_test[client] = [], []
    acc_val_local[client], loss_test_local[client] = [], []
    local_best_acc[client] = 0.01

    nets[client] = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=True).to(DEVICE)
    nets_2[client] = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=True).to(DEVICE)
    nets_history[client] = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=True).to(DEVICE).eval()

    optimizers[client] = optim.AdamW(nets[client].parameters(), lr=LR, weight_decay=WD)
    optimizers_2[client] = optim.AdamW(nets_2[client].parameters(), lr=LR, weight_decay=WD)
print('Finished creating local models.')

print('Init Aggre Weights of each Client:', WEIGHTS_CL)
LEN_UDIAT = len(consistent_path['UDIAT'][FOLD]['x_train'])

print('==== Start training ====')
for epoch in range(EPOCHS):
    index.append(epoch)
    SELECT_DATA = [0, 0, 0, LEN_UDIAT]
    start_time = time.time()

    if epoch < 0.95 *EPOCHS:
        fed_broadcast(CLIENTS, nets, epoch, best_epoch, step2_flag= step2_flag, data_ps=select_data_ratios, super_types=CLIENTS_SUPERVISION, fed_name='global')
        fed_broadcast(CLIENTS_2, nets_2, epoch, best_epoch, step2_flag= step2_flag, data_ps=select_data_ratios, super_types=CLIENTS_SUPERVISION, fed_name='global')


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
        total_data = len(consistent_path[client][FOLD]['x_train'])
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
            local_train_f(training_loader[client], nets_2[client], nets_2['global'], optimizers_2[client], DEVICE,
                          acc=None, loss=None, epoch=epoch,
                          supervision_type=supervision_t, Local_epoch=LOCAL_EPOCH)
        else:
            local_train_w(training_loader[client], nets[client], nets['global'], optimizers[client], DEVICE,
                          acc=acc_train[client], loss=loss_train[client], epoch=epoch,
                          supervision_type=supervision_t, FedMix_network=1, Local_epoch=LOCAL_EPOCH,
                          net_history = nets_history[client])
            local_train_w(training_loader[client], nets_2[client], nets_2['global'], optimizers_2[client], DEVICE,
                          acc=None, loss=None, epoch=epoch,
                          supervision_type=supervision_t, FedMix_network=2, Local_epoch=LOCAL_EPOCH,
                          net_history = nets_history[client])

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


    if epoch < 0.95 * EPOCHS:
        fed_aggr(CLIENTS, WEIGHTS, nets, fed_name='global')
        fed_aggr(CLIENTS_2, WEIGHTS, nets_2, fed_name='global')


    avg_acc = 0.0
    for client in CLIENTS:
        val(epoch, testing_loader[client], nets['global'], DEVICE, acc_val[client],
            loss_test[client])
        avg_acc += acc_val[client][-1]
    avg_acc = avg_acc / TOTAL_CLIENTS


    avg_acc_local = 0.0
    for client in CLIENTS:

        val_local(epoch, client, testing_loader[client], nets[client], DEVICE, acc_val_local[client],
                  loss_test_local[client], nets_history[client], local_best_acc)
        avg_acc_local += acc_val_local[client][-1]
    avg_acc_local = avg_acc_local / TOTAL_CLIENTS

    if check_acc_val(acc_val, 0.50)and step2_flag == False:
        print("===========Step 2============")
        step2_flag = True

    if avg_acc > best_avg_acc:
        best_avg_acc = avg_acc
        best_epoch = epoch
        if best_epoch > 0.7 * EPOCHS:
            save_model(SAVE_MODEL_PATH, best_epoch, nets, acc_train, isbest=True)


    clear_output(wait=True)
    print('Epoch:', epoch, '|', 'Global_acc:', avg_acc, '|','Local_acc:', avg_acc_local, '|' 'Best_global_acc:', best_avg_acc)
    print('Epoch:', epoch, '|', 'acc_train:', acc_train['BMC'][-1], '|',acc_train['BUS'][-1], '|', acc_train['BUSIS'][-1], '|',
          acc_train['UDIAT'][-1])
    print('Epoch:', epoch, '|', 'loss_train:', loss_train['BMC'][-1], '|',loss_train['BUS'][-1], '|', loss_train['BUSIS'][-1], '|',
          loss_train['UDIAT'][-1])
    print('Epoch:', epoch, '|', 'global_test:', acc_val['BMC'][-1], '|', acc_val['BUS'][-1], '|', acc_val['BUSIS'][-1], '|',
          acc_val['UDIAT'][-1])
    print('Epoch:', epoch, '|', 'local_test:', acc_val_local['BMC'][-1], '|', acc_val_local['BUS'][-1], '|', acc_val_local['BUSIS'][-1], '|',
          acc_val_local['UDIAT'][-1])
    print(f"Epoch {epoch} cost {time.time() - start_time :.2f} seconds")
    print("===================================================================")
    

    acc_avg, loss_avg = 0, 0
    save_data_to_file(CLIENTS, acc_val, loss_train, SAVE_LOG_PATH + '/' + LOG_FILE_NAME)



plt.figure(0)
plt.plot(index, acc_train['BMC'], colors[0], label='BMC train',linewidth = linewidth)
plt.plot(index, acc_train['BUS'], colors[1], label='BUS train',linewidth = linewidth)
plt.plot(index, acc_train['BUSIS'], colors[2], label='BUSIS train',linewidth = linewidth)
plt.plot(index, acc_train['UDIAT'], colors[3], label='UDIAT train',linewidth = linewidth)
plt.grid(True)
plt.legend()
plt.savefig(SAVE_LOG_PATH + '/' + 'train_acc_curve.png')
# plt.show()

plt.figure(1)
plt.plot(index, acc_val['BMC'], colors[0], label='BMC test', linewidth = linewidth)
plt.plot(index, acc_val['BUS'], colors[1], label='BUS test', linewidth = linewidth)
plt.plot(index, acc_val['BUSIS'], colors[2], label='BUSIS test', linewidth = linewidth)
plt.plot(index, acc_val['UDIAT'], colors[3], label='UDIAT test', linewidth = linewidth)
plt.plot(index, [(a+b+c+d) / 4 for a,b,c,d in zip(acc_val['UDIAT'], acc_val['BMC'], acc_val['BUS'], acc_val['BUSIS'])],
         colors[4], label='global test', linewidth = 1)
plt.grid(True)
plt.legend()
plt.savefig(SAVE_LOG_PATH + '/' + 'test_acc_curve.png')
# plt.show()

plt.figure(2)
plt.plot(index, acc_val_local['BMC'], colors[0], label='BMC local test', linewidth = linewidth)
plt.plot(index, acc_val_local['BUS'], colors[1], label='BUS local test', linewidth = linewidth)
plt.plot(index, acc_val_local['BUSIS'], colors[2], label='BUSIS local test', linewidth = linewidth)
plt.plot(index, acc_val_local['UDIAT'], colors[3], label='UDIAT local test', linewidth = linewidth)
plt.plot(index, [(a+b+c+d) / 4 for a,b,c,d in zip(acc_val_local['UDIAT'], acc_val_local['BMC'], acc_val_local['BUS'], acc_val_local['BUSIS'])],
         colors[4], label='global test', linewidth = 1)
plt.grid(True)
plt.legend()
plt.savefig(SAVE_LOG_PATH + '/' + 'acc_test_local_curve.png')
# plt.show()


print('Best_glob_acc and epoch:', best_avg_acc, best_epoch)
for client in CLIENTS:
    tmp = best_epoch
    best_epoch = best_epoch
    print(f"##{client}##")
    print("Shared epoch specific:", acc_val[client][best_epoch])
    print("Max client-specific:", np.max(acc_val[client]))
    best_epoch = tmp
print(time.strftime('Finish:%Y-%m-%d %H:%M:%S', time.localtime(time.time())))



