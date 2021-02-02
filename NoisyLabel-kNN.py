# +
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
import pandas as pd
import os, glob, sys, time, math, random
import cv2, pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from numpy.random import choice
import gc

from utils.meter import AverageValueMeter
from utils.MSIDataset import get_knn_dataset, get_train_test_imgs, drop_train_mss
from utils.kNN_epoch import get_train_test_epoch
from utils.loss import CrossEntropyLoss
from utils.metric import fscore
from utils.kNN_cuda import IterKNN

msi_mantis = pd.read_csv('/data/COAD_msimantis_score.csv')
patient_ids = msi_mantis['Patient ID'].tolist()
patient_cls = msi_mantis['MSIMSS'].tolist()
lookup = dict(zip(patient_ids, patient_cls))

tumor_patches = np.load('/data/512dense_mantis_tumor_Zenodo+DrYu.npy')
train_patient, test_patient = train_test_split(patient_ids, test_size = 0.2, random_state = 42)

class Resnet34(nn.Module):
    def __init__(self,):
        super().__init__()
        resnet = models.resnet34() 
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.classifer = nn.Linear(512, 2)
    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()

        logit = self.classifer(h)
        return logit, h


def adjust_learning_rate(init_lr, optimizer, epoch):
    lr = init_lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_model():
    model = Resnet34()
    return model

def main():
    train_imgs, valid_imgs = get_train_test_imgs(tumor_patches, train_patient), get_train_test_imgs(tumor_patches, test_patient)
    train_imgs = drop_train_mss(train_imgs, lookup)
    
    patch_int_lookup = {}
    int_patch_lookup = {}
    patch_cls_lookup = {}
    for i, p in enumerate(np.concatenate((train_imgs, valid_imgs))):
        patch_id = p.split('/')[-1][:-4]
        patch_int_lookup[patch_id] = i
        int_patch_lookup[i] = patch_id
        p_id = p.split('/')[-1][:12]
        patch_cls_lookup[i] = lookup[p_id]
    train_image_int = [patch_int_lookup[p.split('/')[-1][:-4]] for p in train_imgs]
    
    train_dataset, extract_dataset, valid_dataset = get_knn_dataset(train_imgs, valid_imgs, patch_cls_lookup, patch_int_lookup)
    
    max_epoch = 30
    lr = 2e-3
    weight_decay=2e-3
    momentum=0.9
    batch_size = 96
    DEVICE = 'cuda:0'
    
    init_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    extract_loader = DataLoader(extract_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
    
    model = get_model()
    
    criterion = CrossEntropyLoss()
    metrics = [fscore()]
    optimizer = torch.optim.SGD([ 
        dict(params=model.parameters(), lr=lr),
    ], weight_decay=weight_decay,  momentum=momentum, nesterov=True)
    
    train_epoch, extract_epoch, valid_epoch = get_train_test_epoch(model, criterion, metrics, optimizer, DEVICE, int_patch_lookup, lookup)
    
    iterknn = IterKNN()
    agerage_pred = {k : train_dataset.knn_lookup[k] for k in train_image_int}
    pseudo_label = None
    
    max_score = 0
    
    for e in range(1, max_epoch + 1):

        print('\nEpisode : {}'.format(e))
        if(e == 1):
            logs = {}
            loss_meter = AverageValueMeter()
            with tqdm(init_train_loader, desc='init train', file=sys.stdout) as iterator:
                for x, _, ori_y in iterator:
                    x, ori_y = x.to(DEVICE), ori_y.to(DEVICE)

                    optimizer.zero_grad()
                    prediction, feat = model.forward(x)
                    del feat

                    l = criterion(prediction, ori_y).mean()
                    l.backward()
                    optimizer.step()

                    # update loss logs
                    loss_value = l.cpu().detach().numpy()
                    loss_meter.add(loss_value)
                    loss_logs = {'CELoss': loss_meter.mean}
                    logs.update(loss_logs)

                    iterator.set_postfix_str('CELoss:{}'.format(logs['CELoss']))

            del init_train_loader        
            gc.collect()

        else :

            embedding, avgerage_pred = extract_epoch.run(extract_loader, avgerage_pred)
            pseudo_label = iterknn(embedding, avgerage_pred)        
            del embedding
            gc.collect()
            train_dataset.knn_lookup.update(pseudo_label)

            for i in range(1, max_epoch + 1):
                train_logs = train_epoch.run(train_loader, gamma)

            gc.collect()

        adjust_learning_rate(lr, optimizer, e)
        valid_logs = valid_epoch.run(test_loader)

        gc.collect()
        
if __name__ == "__main__":
    main()
