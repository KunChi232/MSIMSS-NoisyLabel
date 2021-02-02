import torch
import torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
import pandas as pd
import albumentations as albu
import os, glob, sys
import cv2, pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils.meter import AverageValueMeter
from utils.MSIDataset import get_patchbased_dataset, get_train_test_imgs, drop_train_mss
from utils.patch_based_epoch import get_train_test_epoch
from utils.loss import CrossEntropyLoss
from utils.metric import fscore



msi_mantis = pd.read_csv('/data/COAD_msimantis_score.csv')
patient_ids = msi_mantis['Patient ID'].tolist()
patient_cls = msi_mantis['MSIMSS'].tolist()
lookup = dict(zip(patient_ids, patient_cls))

tumor_patches = np.load('/data/512dense_mantis_tumor_Zenodo+DrYu.npy')
train_patient, test_patient = train_test_split(patient_ids, test_size = 0.2, random_state = 42)

def get_model():
    model = models.resnet34()
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

# +
def main():
    train_imgs, valid_image = get_train_test_imgs(tumor_patches, train_patient), get_train_test_imgs(tumor_patches, test_patient)
    train_imgs = drop_train_mss(train_imgs, lookup)
    train_dataset, valid_dataset = get_patchbased_dataset(train_imgs, valid_image, lookup)
    
    max_epoch = 30
    lr = 2e-3
    weight_decay=2e-3
    momentum=0.9
    batch_size = 128
    DEVICE = 'cuda:0'
    
    train_loader = DataLoader(train_dataset, num_workers=16, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    
    model = get_model()
    
    criterion = CrossEntropyLoss()
    metrics = [fscore()]
    optimizer = torch.optim.SGD([ 
        dict(params=model.parameters(), lr=lr),
    ], weight_decay=weight_decay,  momentum=momentum, nesterov=True)
    
    train_epoch, valid_epoch = get_train_test_epoch(model, criterion, metrics, optimizer, DEVICE, lookup)
    
    #model_name = '/data/msimss_tcga/weight/'+ str(current_time) + "Resnet34-MSI-MSS-Classification_bs:{}".format(batch_size)

    max_score = 0
    for i in range(1, max_epoch + 1):
        print('\nEpoch: {}, batch: {}'.format(i, batch_size))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
 

        if max_score < valid_logs['AUROC']:
            max_score = valid_logs['AUROC']
            
if __name__ == "__main__":
    main()
