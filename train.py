import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import time, argparse
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from model import PyramidFlow
from util import MOAIDataloader, BatchDiffLoss
from util import fix_randseed, compute_pro_score_fast, getLogger

import csv
from pathfilemgr import MPathFileManager
from hyp_data import MHyp
from utiles import remaining_time

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--volume', help='volume directory', default='moai')
parser.add_argument('--project', help='project directory', default='test_project')
parser.add_argument('--subproject', help='subproject directory', default='test_subproject')
parser.add_argument('--task', help='task directory', default='test_task')
parser.add_argument('--version', help='version', default='v1')

opt = parser.parse_args()

##########################################################################################
### make dir and hyp
##########################################################################################

mpfm = MPathFileManager(opt.volume, opt.project, opt.subproject, opt.task, opt.version)
mhyp = MHyp()
mpfm.load_train_hyp(mhyp)

##########################################################################################
### make results.csv header
##########################################################################################

result_file = open(mpfm.result_csv, mode='a', newline='', encoding='utf-8')
result_csv = csv.writer(result_file)
result_csv.writerow(["epoch", "time", "pixel_auroc", "image_auroc", "pixel_pro", "loss"])

##########################################################################################
### tensorboard
##########################################################################################

twriter = SummaryWriter(mpfm.train_result)

##########################################################################################
### save arguments
##########################################################################################

mpfm.save_hyp(mhyp)


def train(train_path, val_path, resnetX, num_layer, vn_dims, ksize, channel, num_stack, device, batch_size, epoches, hyp):
    # save config
    save_dict = {'cls_name': 'moai', 'resnetX': resnetX, 'num_layer': num_layer, 'vn_dims': vn_dims,\
                 'ksize': ksize, 'channel': channel, 'num_stack': num_stack, 'batch_size': batch_size}
    loader_dict = fix_randseed(seed=0)

    # model
    flow = PyramidFlow(resnetX, channel, num_layer, num_stack, ksize, vn_dims, False).to(device)
    x_size = 256 if resnetX==0 else 1024
    # optimizer = torch.optim.Adam(flow.parameters(), lr=2e-4, eps=1e-04, weight_decay=1e-5, betas=(0.5, 0.9)) # using cs-flow optimizer
    optimizer = torch.optim.AdamW(flow.parameters(), lr=2e-4, eps=1e-04, weight_decay=1e-5, betas=(0.5, 0.9))
    Loss = BatchDiffLoss(batch_size, p=2)

    # dataset
    train_dataset = MOAIDataloader(mode='train', x_size=x_size, y_size=256, datapath=train_path, hyp=hyp)
    val_dataset = MOAIDataloader(mode='val', x_size=x_size, y_size=256, datapath=train_path)
    test_dataset = MOAIDataloader(mode='test', x_size=x_size, y_size=256, datapath=val_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, persistent_workers=False, pin_memory=True, drop_last=True, **loader_dict)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, persistent_workers=False, pin_memory=True, drop_last=False, **loader_dict)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, persistent_workers=False, pin_memory=True, **loader_dict)

    # training & evaluation
    pixel_auroc_lst = [0]
    pixel_pro_lst = [0]
    image_auroc_lst = [0]
    losses_lst = [0]
    t0 = time.time()
    for epoch in range(epoches):
        # train
        flow.train()
        losses = []
        for train_dict in train_loader:
            image, labels = train_dict['images'].to(device), train_dict['labels'].to(device)
            optimizer.zero_grad()
            pyramid2= flow(image)
            diffes = Loss(pyramid2)
            diff_pixel = flow.pyramid.compose_pyramid(diffes).mean(1)  
            loss = torch.fft.fft2(diff_pixel).abs().mean() # Fourier loss
            loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1e0) # Avoiding numerical explosions
            optimizer.step()
            losses.append(loss.item())
        mean_loss = np.mean(losses)
        losses_lst.append(mean_loss)

        # val for template 
        flow.eval()
        feat_sum, cnt = [0 for _ in range(num_layer)], 0
        for val_dict in val_loader:
            image = val_dict['images'].to(device)
            with torch.no_grad():
                pyramid2= flow(image) 
                cnt += 1
            feat_sum = [p0+p for p0, p in zip(feat_sum, pyramid2)]
        feat_mean = [p/cnt for p in feat_sum]

        # test
        flow.eval()
        diff_list, labels_list = [], []
        for test_dict in test_loader:
            image, labels, fname = test_dict['images'].to(device), test_dict['labels'], test_dict['fname']
            with torch.no_grad():
                pyramid2 = flow(image) 
                pyramid_diff = [(feat2 - template).abs() for feat2, template in zip(pyramid2, feat_mean)]
                diff = flow.pyramid.compose_pyramid(pyramid_diff).mean(1, keepdim=True)# b,1,h,w
                diff_list.append(diff.cpu())
                labels_list.append(labels.cpu()==1)# b,1,h,w

                diff = np.squeeze(diff.cpu().numpy())
                amap_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
                plt.imshow(amap_norm, cmap='jet')
                plt.axis('off')
                plt.savefig(f'{mpfm.train_result}/{epoch}_{fname[0]}.png', bbox_inches='tight', pad_inches=0)



        labels_all = torch.concat(labels_list, dim=0)# b1hw 
        amaps = torch.concat(diff_list, dim=0)# b1hw 
        amaps, labels_all = amaps[:, 0], labels_all[:, 0] # both b,h,w
        pixel_auroc = roc_auc_score(labels_all.flatten(), amaps.flatten()) # pixel score
        image_auroc = roc_auc_score(labels_all.amax((-1,-2)), amaps.amax((-1,-2))) # image score
        pixel_pro = compute_pro_score_fast(amaps, labels_all) # pro score

        if pixel_auroc > np.max(pixel_auroc_lst):
            save_dict['state_dict_pixel'] = {k: v.cpu() for k, v in flow.state_dict().items()} # save ckpt
        if pixel_pro > np.max(pixel_pro_lst):
            save_dict['state_dict_pro'] = {k: v.cpu() for k, v in flow.state_dict().items()} # save ckpt
        pixel_auroc_lst.append(pixel_auroc)
        pixel_pro_lst.append(pixel_pro)
        image_auroc_lst.append(image_auroc)

        # MOAI Logs
        twriter.add_scalar("train/loss", mean_loss, epoch)
        remaining = remaining_time(t0, epoch, epoches)
        result_csv.writerow([epoch, remaining, pixel_auroc, image_auroc, pixel_pro, mean_loss])
        result_file.flush()
        
        # print
        print(f'[epoch: {epoch}/{epoches}] - remaining time: {remaining}s')
        print(f'   Loss: {mean_loss}')
        print(f'   Pixel-AUROC: {pixel_auroc}')
        print(f'   Image-AUROC: {image_auroc}')
        print(f'   Pixel-PRO: {pixel_pro}')

        del amaps, labels_all, diff_list, labels_list

    save_dict['pixel_auroc_lst'] = pixel_auroc_lst
    save_dict['image_auroc_lst'] = image_auroc_lst
    save_dict['pixel_pro_lst']   = pixel_pro_lst
    save_dict['losses_lst'] = losses_lst

    np.savez(f'{mpfm.weight_path}/best.npz', **save_dict) # save all 

    result_file.close()

if __name__ == '__main__':

    if mhyp.volumeNorm == 'CVN':
        vn_dims = (0, 1)
    elif mhyp.volumeNorm == 'SVN':
        vn_dims = (0, 2, 3)

    resnetX = 0 if mhyp.encoder=='none' else int(mhyp.encoder[6:])

    train(mpfm.train_path, mpfm.val_path,\
          resnetX, mhyp.numLayer, vn_dims, \
          ksize=mhyp.kernelSize, channel=mhyp.numChannel, num_stack=mhyp.numStack, \
          device=f'cuda:{mhyp.gpu}', batch_size=mhyp.batchSize, epoches=mhyp.epoches, hyp=mhyp)

