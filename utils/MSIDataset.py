import numpy as np
import pandas as pd
import random
import cv2
import albumentations as albu
from torch.utils.data import DataLoader, Dataset, Sampler
from sklearn.model_selection import train_test_split


# msi_mantis = pd.read_csv('./COAD_msimantis_score.csv')
# patient_ids = msi_mantis['Patient ID'].tolist()
# patient_cls = msi_mantis['MSIMSS'].tolist()
# lookup = dict(zip(patient_ids, patient_cls))

# tumor_patches = np.load('../data/512dense_mantis_tumor_Zenodo+DrYu.npy')

# train_patient, test_patient = train_test_split(patient_ids, test_size = 0.2, random_state = 42)

def get_train_test_imgs(tumor_patches, patient):
    imgs = []
    for p in tumor_patches:
        p_id = p.split('/')[-1][:12]
        if(p.split('/')[-1][13] == '1'):
            continue
        if(p_id in patient):
            imgs.append(p)
    return imgs


def drop_train_mss(train_imgs, lookup):
    random.seed(42)
    train_mss = [k for k in train_imgs if lookup[k.split('/')[-1][:12]] == 0 ]
    drop = random.sample(train_mss, 970292 - 650000)
    drop = set(drop)
    train_imgs = [k for k in train_imgs if k not in drop]
    return train_imgs



def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing():
    _transform = [
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def get_training_augmentation():
    test_transform = [
        albu.Rotate((90, 170, 280), border_mode=0, value=0),
        albu.Flip(),
        albu.RandomResizedCrop(224, 224, scale=(0.4, 1.0)),
        albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0),
        albu.ColorJitter(p = 0.8),
        albu.OneOf([
            albu.GaussianBlur(sigma_limit=(0.1, 2.0)),
            albu.IAASharpen()
        ], p=0.5),
    ]
    return albu.Compose(test_transform)

def get_validation_augmentation():
    test_transform = [
        albu.Resize(224, 224)
        #albu.ToGray(p = 1.)
    ]
    return albu.Compose(test_transform)

class PatchBasedDataset(Dataset):
    def __init__(self, imgs, lookup, augmentation, preprocessing, validation = False):
        self.imgs = imgs
        self.lookup = lookup
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.validation = validation
    def __getitem__(self, i):
        file_name = self.imgs[i]
        patch_name = file_name.split('/')[-1][:-4] 
        p_id = patch_name[:12]
        img = cv2.imread(file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if(self.augmentation):
            sampled = self.augmentation(image = img)
            img = sampled['image']
        if(self.preprocessing):
            sampled = self.preprocessing(image = img)
            img = sampled['image']
        if(self.validation):
            return img / 255., self.lookup[p_id], patch_name
        else:
            return img / 255., self.lookup[p_id]
    
    def __len__(self):
        return len(self.imgs)

class KNNDataset(Dataset):
    def __init__(self, imgs, lookup, patch_int_lookup, augmentation, preprocessing, validation = False):
        self.imgs = imgs
        self.ori_lookup = lookup.copy()
        self.patch_int_lookup = patch_int_lookup
        self.knn_lookup = {k : self.onehot(lookup[k]) for k in lookup.keys()}
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.validation = validation
    def onehot(self, c):
        if(c == 1):
            return np.array([0., 1.])
        else:
            return np.array([1., 0.])
    def __getitem__(self, i):
        file_name = self.imgs[i]
        patch_name = file_name.split('/')[-1][:-4]
        patch_int = self.patch_int_lookup[patch_name]
        img = cv2.imread(file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if(self.augmentation):
            sampled = self.augmentation(image = img)
            img = sampled['image']
        if(self.preprocessing):
            sampled = self.preprocessing(image = img)
            img = sampled['image']
            
        if(self.validation):
            return img / 255.0, np.array(self.ori_lookup[patch_int]), np.array(patch_int)
        else:
            return img / 255.0, np.array(self.knn_lookup[patch_int]), np.array(self.ori_lookup[patch_int])
    
    def __len__(self):
        return len(self.imgs)    

def get_patchbased_dataset(train_imgs, test_imgs, lookup):
    train_dataset = PatchBasedDataset(
        imgs = train_imgs,
        lookup = lookup,
        augmentation = get_training_augmentation(),
        preprocessing = get_preprocessing(),
        validation=False
    )

    validation_dataset = PatchBasedDataset(
        imgs = test_imgs,
        lookup = lookup,
        augmentation = get_validation_augmentation(),
        preprocessing = get_preprocessing(),
        validation = True
    ) 
    
    return train_dataset, validation_dataset

def get_knn_dataset(train_imgs, test_imgs, patch_cls_lookup, patch_int_lookup):
    train_dataset = KNNDataset(
        imgs = train_imgs,
        lookup = patch_cls_lookup,
        patch_int_lookup = patch_int_lookup,
        augmentation = get_training_augmentation(),
        preprocessing = get_preprocessing(),
        validation=False
    )

    extract_dataset = KNNDataset(
        imgs = train_imgs,
        lookup = patch_cls_lookup,
        patch_int_lookup = patch_int_lookup,
        augmentation = get_validation_augmentation(),
        preprocessing = get_preprocessing(),
        validation = True
    ) 
    
    validation_dataset = KNNDataset(
        imgs = test_imgs,
        lookup = patch_cls_lookup,
        patch_int_lookup = patch_int_lookup,
        augmentation = get_validation_augmentation(),
        preprocessing = get_preprocessing(),
        validation = True
    )    
    return train_dataset, extract_dataset, validation_dataset
