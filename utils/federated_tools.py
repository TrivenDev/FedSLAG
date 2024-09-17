import math
import os
import time

import numpy as np
import torchvision.transforms.functional as TF
import copy

from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import Resize, RandomCrop
from torch.utils.data import Dataset


from dice_loss import dice_coeff
from configs.breast.config import EPOCHS as TOTAL_EPOCHS

import random
from utils.LGE.tools import *
from utils.mask_to_keypoints import mask2point
from utils.mask_to_scribbles import mask2scribble
from utils.mask_to_bbox import mask2bbox

colors = ['r', 'g', 'b', 'c', 'k', 'y', 'm', 'c']

class Cancer(Dataset):
    def __init__(self, im_path, mask_path, train=False, \
                 IMAGE_SIZE=(256, 256), CROP_SIZE=(224, 224),
                 noisy=True):
        self.data = im_path
        self.label = mask_path
        self.train = train
        self.IMAGE_SIZE = IMAGE_SIZE
        self.CROP_SIZE = CROP_SIZE
        self.noisy = noisy
        self.ignore_class = 1
    
    def get_cls_label(self, cls_label):
        cls_label_set = list(cls_label)

        if self.ignore_class in cls_label_set:
            cls_label_set.remove(self.ignore_class)
        if 255 in cls_label_set:
            cls_label_set.remove(255)


        cls_label = np.zeros(self.ignore_class)
       
        for i in cls_label_set:
            cls_label[i] += 1
        cls_label = torch.from_numpy(cls_label).float()
        return cls_label

    def transform(self, image, mask, train):
        resize_image = Resize(self.IMAGE_SIZE)
        resize_label = Resize(self.IMAGE_SIZE, interpolation=TF.InterpolationMode.NEAREST)
        image = resize_image(image)
        mask = resize_label(mask)

        if train:

            i, j, h, w = RandomCrop.get_params(
                image, output_size=(self.CROP_SIZE))

            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx]).convert('L')
        mask = Image.open(self.label[idx]).convert('L')

        x, y = self.transform(image, mask, self.train)
        y_copy = copy.deepcopy(y)

        y_weak = torch.zeros(y_copy.shape)

        if torch.sum(y_copy) > 0:

            y_weak = mask2bbox(y_copy, noisy=self.noisy)

        return x, y, y_weak

class cancer_v2(Dataset):
    def __init__(self, im_store, y_store, pl1_store, pl2_store, super_type):
        self.im_store = im_store
        self.y_store = y_store
        self.pl1_store = pl1_store
        self.pl2_store = pl2_store
        self.super_type = super_type
        self.ignore_class = 0

    def get_cls_label(self, cls_label):
        if len(cls_label) != 0:
            print('cls label:',cls_label)

        cls_label_set = list(cls_label)
        if len(cls_label_set)==0:
            print("cls label set is empty list")

        if self.ignore_class in cls_label_set:
            cls_label_set.remove(self.ignore_class)
        if 255 in cls_label_set:
            cls_label_set.remove(255)

        cls_label = np.zeros(1)

        if len(cls_label_set)==0:
            print("cls label set is empty list")
        else:
            for i in cls_label_set:
                cls_label[0] += 1
            print(cls_label)

        cls_label = torch.from_numpy(cls_label).float()
        return cls_label

    def __len__(self):
        return len(self.im_store)

    def __getitem__(self, idx):
        x, y, y1, y2 = self.im_store[idx], self.y_store[idx], self.pl1_store[idx], self.pl2_store[idx]
        y_weak = torch.zeros(y.shape)

        if torch.sum(y) > 0:
            if self.super_type == 'point':
                y_weak = mask2point(y)
            elif self.super_type == 'scribble':
                y_weak = mask2scribble(y)
            elif self.super_type == 'bbox':
                y_weak = mask2bbox(y, noisy=True)

        if self.super_type == 'point' or self.super_type == 'scribble':
            cls_label = np.unique(np.asarray(y_weak))
            if 1 in cls_label:
                cls_label = torch.from_numpy(1*np.ones(1)).float()
            else:
                cls_label = torch.from_numpy(0 * np.ones(1)).float()
        else:
            cls_label = torch.from_numpy(-1*np.ones(1)).float()

        return x, y1, y2, cls_label, y_weak


def fed_aggr(CLIENTS, WEIGHTS_CL, nets, fed_name='global'):
    nets[fed_name] = nets[fed_name].to('cuda:0')
    
    for param_tensor in nets[fed_name].state_dict():
        tmp = None
        TOTAL_CLIENTS = len(CLIENTS)
        for client, w in zip(CLIENTS, WEIGHTS_CL):
            nets[client]=nets[client].to('cuda:0')
            if tmp == None:
                tmp = copy.deepcopy(w * nets[client].state_dict()[param_tensor])
            else:
                tmp += w * nets[client].state_dict()[param_tensor]
            
            nets[client]=nets[client].to('cpu')
        
        nets[fed_name].state_dict()[param_tensor].data.copy_(tmp)
        del tmp

def select_pl(nets_1, nets_2, device, trainloader, im_store, y_store, \
              pl1_store, pl2_store, TH=0.9, supervision='labeled'):
    counter, dice_acc = 0, 0
    nets_1= nets_1.eval().to(device)
    nets_2 = nets_2.eval().to(device)


    with torch.no_grad():
        for (imgs, masks, mask_2) in trainloader:
            
            imgs_cuda1, imgs_cuda2 = imgs.to(device), imgs.to(device)


            y1_pred, y2_pred = nets_1(imgs_cuda1), nets_2(imgs_cuda2)
            y1_pred, y2_pred = torch.sigmoid(y1_pred), torch.sigmoid(y2_pred)
            y1_pred, y2_pred = (y1_pred > 0.5).float(), (y2_pred > 0.5).float()

            if supervision=='bbox':
                mask_2 = mask_2.to(device)
                y1_pred_refine = y1_pred * mask_2
                y2_pred_refine = y2_pred * mask_2

            if supervision=='bbox':
                dice_net12 = dice_coeff(y2_pred_refine, y1_pred_refine)
            else:
                dice_net12 = dice_coeff(y2_pred, y1_pred)

            dice_wrt_gt = dice_coeff(masks.type(torch.float).to(device), y1_pred)

            if dice_net12 >= TH:
                dice_acc += dice_wrt_gt
                if supervision =='bbox':

                    dice_bbox = dice_coeff(y1_pred_refine, mask_2)
                    if dice_bbox < 0.5:
                        continue

                im_store.append(imgs[0])
                y_store.append(masks[0])

                counter += 1
                if supervision =='bbox':
                    y1_pred = y1_pred_refine
                    y2_pred = y2_pred_refine

                pl1_store.append(y1_pred[0].detach().cpu())
                pl2_store.append(y2_pred[0].detach().cpu())

    return counter, dice_acc

def val(epoch, testloader, net, device, acc=None, loss=None):
    net.eval()
    net = net.to(device)

    t_loss, t_acc = 0, 0
    with torch.no_grad():
        for (imgs, masks, _) in testloader:
            masks = masks.type(torch.float32)
            imgs, masks = imgs.to(device), masks.to(device)

            masks_pred = net(imgs)
            masks_pred = torch.sigmoid(masks_pred)
            l_ = 1 - dice_coeff(masks_pred, masks.type(torch.float))
            t_loss += l_.item()

            masks_pred = (masks_pred > 0.5).float()
            t_acc_network = dice_coeff(masks.type(torch.float), masks_pred).item()
            t_acc += t_acc_network
   
        new_acc = t_acc / len(testloader)
        new_loss = t_loss / len(testloader)


        if acc is not None:
            acc.append(new_acc)
        if loss is not None:
            loss.append(new_loss)

        del t_acc, t_loss


def val_local(epoch, client, testloader, net, device, acc=None, loss=None, net_history=None, local_best_acc=None):

    net.eval()
    net = net.to(device)
    if net_history: 
        net_history.eval()
        net_history = net_history.to(device)
    t_loss, t_acc = 0, 0

    with torch.no_grad():
        for (imgs, masks, _) in testloader:
            masks = masks.type(torch.float32)
            imgs, masks = imgs.to(device), masks.to(device)

            masks_pred = net(imgs)
            masks_pred = torch.sigmoid(masks_pred)
            l_ = 1 - dice_coeff(masks_pred, masks.type(torch.float))
            t_loss += l_.item()
            masks_pred = (masks_pred > 0.5).float()
            t_acc_network = dice_coeff(masks.type(torch.float), masks_pred).item()
            t_acc += t_acc_network
   
        new_acc = t_acc / len(testloader)
        new_loss = t_loss / len(testloader)


        if net_history and new_acc > local_best_acc[client]:
            local_best_acc[client] = new_acc
            net_history.load_state_dict(copy.deepcopy(net.state_dict()))
            print('Get local best model at:', new_acc)

        if acc is not None:
            acc.append(new_acc)
        if loss is not None:
            loss.append(new_loss)

        del t_acc, t_loss


def local_train_w(trainloader, net_stu, net_global, optimizer_stu, \
                  device, acc=None, loss=None, epoch=1, supervision_type='bbox', \
                  warmup=False, CE_LOSS=None, FedMix_network=1, Local_epoch=1, net_history=None):
    net_stu.train()
    net_stu = net_stu.to(device)

    criterion = nn.KLDivLoss()

    t_loss, t_acc = 0.0, 0.0
    l_ = 0
    diff_weight, gmm_weight = 0.2*(epoch/TOTAL_EPOCHS), 0.5 * math.exp(-epoch / (1.0 * TOTAL_EPOCHS))
    seg_weight = 1-diff_weight-gmm_weight
    PROXIMAL_STRENGTH = 1e-3


    for local_epoch in range(Local_epoch):
        cls_labels, y_weak = None, None
        labeled_len = len(trainloader)
        labeled_iter = iter(trainloader) 

        for _ in range(labeled_len):
            valid_gmm = True
            try:
                iter_tmp = next(labeled_iter)
                imgs, masks, masks2, cls_labels, y_weak = iter_tmp
                imgs, masks, cls_labels = imgs.to(device), masks.to(device), cls_labels.to(device)
            except ValueError as e:
                try:
                    imgs, masks, masks2 = iter_tmp[:3]
                    imgs, masks = imgs.to(device), masks.to(device)
                except ValueError as e:
                    print(f"Still encountered Dataloader error: {e}")
                    continue

            l_ = 0
            loss_seg, loss_gmm, loss_diff = 0,0,0

            feat, pred_stu = net_stu(imgs)
            masks_stu = torch.sigmoid(pred_stu)
            feat = torch.sigmoid(feat)


            if supervision_type == 'labeled':
                l_stu = (1 - dice_coeff(masks_stu, masks.type(torch.float)))[0]
                loss_seg = 0.99*l_stu
            elif supervision_type in ['unlabeled','bbox']:
                if FedMix_network == 1:
                    masks_teach = masks2.to(device)
                else:
                    masks_teach = masks.to(device)
                l_stu = (1 - dice_coeff(masks_stu, masks_teach.type(torch.float)))[0]
                loss_seg = 0.99*l_stu
            else:
                if FedMix_network == 1:
                    masks_teach = masks2.to(device)
                else:
                    masks_teach = masks.to(device)
                l_stu = (1 - dice_coeff(masks_stu, masks_teach.type(torch.float)))[0]
                loss_seg = seg_weight* l_stu

            if net_history:
                net_history = net_history.to(device)
                with torch.no_grad():
                    teacher_outputs = net_history(imgs)
                student_outputs = pred_stu
                diffusion_loss = criterion(F.log_softmax(student_outputs, dim=-1), F.softmax(teacher_outputs, dim=-1))
                loss_diff += diff_weight * diffusion_loss

                net_global = net_global.to(device)

            if y_weak is not None and supervision_type in ['point', 'scribble']:
                nclass = 1
                b,_,h,w = masks_stu.size()
                y_weak = y_weak.to(device)
                cur_cls_label = build_cur_cls_label(y_weak, nclass)
                vecs, proto_loss = cal_protypes(feat, y_weak, nclass)
                if torch.isnan(proto_loss):
                    print("proto_loss is NaN. Skip update.")
                    valid_gmm = False  # 设置标志变量，跳过后续计算
                else:
                    res = GMM(feat, vecs, masks_stu, y_weak, cur_cls_label)
                    gmm_loss = cal_gmm_loss(masks_stu, res, cur_cls_label, y_weak) + proto_loss
                    if torch.isnan(gmm_loss):
                        print("gmm_loss is NaN. Skip update.")
                        valid_gmm = False
                    else:
                        loss_gmm += gmm_weight * gmm_loss.view_as(l_stu)

            global_params,local_params = net_global.state_dict(), net_stu.state_dict()
            proximal_term = 0.0
            for param_name in global_params:
                diff = global_params[param_name] - local_params[param_name]
                proximal_term += torch.norm(diff) ** 2
            loss_prox = 0.5 * PROXIMAL_STRENGTH * proximal_term

            l_ = loss_seg + loss_diff + loss_prox
            if valid_gmm:
                l_ += loss_gmm

            optimizer_stu.zero_grad()
            l_.backward()
            optimizer_stu.step()
            t_loss += l_.item()

            masks_stu = (masks_stu.detach() > 0.5).float()
            t_acc_network = dice_coeff(masks_stu, masks.type(torch.float)).item()
            t_acc += t_acc_network


    if acc is not None:
        try:
            acc.append(t_acc/ len(trainloader))
        except:
            print('acc append 0')
            acc.append(0.0)

    if loss is not None:
        try:   
            loss.append(t_loss/ len(trainloader))
        except:
            print('loss append 0')
            loss.append(0.0)


def local_train_f(trainloader, net_stu, net_global, optimizer_stu, \
                  device, acc=None, loss=None, epoch=1, supervision_type='labeled', \
                  warmup=False, CE_LOSS=None, FedMix_network=1, Local_epoch=1, net_history=None):
    net_stu.train()
    net_stu=net_stu.to(device)
    net_global=net_global.to(device)
    criterion = nn.KLDivLoss()

    t_loss, t_acc = 0, 0
    l_ = 0
    diff_weight = 0.2*epoch / TOTAL_EPOCHS
    seg_weight = 1-diff_weight
    PROXIMAL_STRENGTH = 1e-3


    for local_epoch in range(Local_epoch):
        labeled_len = len(trainloader)
        labeled_iter = iter(trainloader)

        # 遍历所有数据样本
        for _ in range(labeled_len):
            imgs, masks, y_pl= next(labeled_iter)
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer_stu.zero_grad()
            l_ = 0

            if FedMix_network == 2:
                imgs = add_gaussian_noise(imgs)

            feat, pred_stu = net_stu(imgs)
            masks_stu = torch.sigmoid(pred_stu)


            if supervision_type == 'labeled':
                l_stu = (1 - dice_coeff(masks_stu, masks.type(torch.float)))[0]
                l_ = seg_weight * l_stu
            

            global_params = net_global.state_dict()
            local_params = net_stu.state_dict()
            proximal_term = 0.0
            for param_name in global_params:
                diff = global_params[param_name] - local_params[param_name]
                proximal_term += torch.norm(diff) ** 2
            proximal_term *= PROXIMAL_STRENGTH / 2.0

            l_ +=  proximal_term

            if net_history:

                net_history = net_history.to(device)
                with torch.no_grad():
                    teacher_outputs = net_history(imgs)
                student_outputs = pred_stu
                diffusion_loss = criterion(F.log_softmax(student_outputs, dim=1), F.softmax(teacher_outputs, dim=1))

                l_ += diff_weight * diffusion_loss

                net_global = net_global.to(device)

            l_.backward()
            optimizer_stu.step()



            t_loss += l_.item()
            masks_stu = (masks_stu.detach() >= 0.5).float()
            t_acc_network = dice_coeff(masks_stu, masks.type(torch.float)).item()
            t_acc += t_acc_network


    if acc is not None:
        try:
            acc.append(t_acc / len(trainloader))
        except:
            print('acc append 0')
            acc.append(0.0)
    if loss is not None:
        try:
            loss.append(t_loss / len(trainloader))
        except:
            print('loss append 0')
            loss.append(0.0)



def add_gaussian_noise(image_tensor, mean=0, std=0.8):

    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("Input image must be a PyTorch tensor.")

    image_tensor = image_tensor.float()
    noise_tensor = torch.randn_like(image_tensor, dtype=torch.float32) * std + mean
    noisy_image_tensor = image_tensor + noise_tensor
    noisy_image_tensor = torch.clamp(noisy_image_tensor, 0, 255)

    if image_tensor.dtype == torch.uint8:
        noisy_image_tensor = noisy_image_tensor.byte()

    return noisy_image_tensor
