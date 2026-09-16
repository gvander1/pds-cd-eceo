import os
import torch
from torch.utils.data import Dataset
#import yaml
from PIL import Image
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
import pickle
import torch.nn.functional as F

DATA_FOLDER = os.getenv("BCD_DATA_FOLDER", "/data/gaetane")
FOLDER_TRAIN_I1 = os.path.join(DATA_FOLDER, 'HiUCD_mini/train/image/2017/9')
FOLDER_TRAIN_I2 = os.path.join(DATA_FOLDER, 'HiUCD_mini/train/image/2018/9')
FOLDER_TRAIN_CD = os.path.join(DATA_FOLDER, 'HiUCD_mini/train/mask_merge/2017_2018/9')

FOLDER_TEST_I1 = os.path.join(DATA_FOLDER, 'HiUCD_mini/test/image/2018/9')
FOLDER_TEST_I2 = os.path.join(DATA_FOLDER, 'HiUCD_mini/test/image/2019/9')
FOLDER_TEST_CD = os.path.join(DATA_FOLDER, 'HiUCD_mini/test/mask_merge/2018_2019/9')

FOLDER_VAL_I1 = os.path.join(DATA_FOLDER, 'HiUCD_mini/val/image/2017/9')
FOLDER_VAL_I2 = os.path.join(DATA_FOLDER, 'HiUCD_mini/val/image/2018/9')
FOLDER_VAL_CD = os.path.join(DATA_FOLDER, 'HiUCD_mini/val/mask_merge/2017_2018/9')


class HiUCD_Dataset(Dataset):
    #cropsize modidié à 224 pour que ce soit un multiple de 14 pour l'encoder DINOv2
    def __init__(self, cropsize=224, type = 'train', small=False, size=10, everything=False, semantic=True, get_name=False):
        self.get_name =get_name

        if type == 'train':
            folderpath_i1 = FOLDER_TRAIN_I1
            folderpath_i2 = FOLDER_TRAIN_I2
            folderpath_change = FOLDER_TRAIN_CD

        if type == 'test':
            folderpath_i1 = FOLDER_TEST_I1
            folderpath_i2 = FOLDER_TEST_I2
            folderpath_change = FOLDER_TEST_CD

        if type == 'val':
            folderpath_i1 = FOLDER_VAL_I1
            folderpath_i2 = FOLDER_VAL_I2
            folderpath_change = FOLDER_VAL_CD


        self.im1 = [os.path.join(folderpath_i1,im) for im in sorted(os.listdir(folderpath_i1)) if im.endswith(".png")] 
        self.im2 = [os.path.join(folderpath_i2,im) for im in sorted(os.listdir(folderpath_i2)) if im.endswith(".png")] 
        self.lab = [os.path.join(folderpath_change,im) for im in sorted(os.listdir(folderpath_change)) if im.endswith(".png")]

        self.type = type
        self.cropsize=cropsize
        self.everything = everything
        self.semantic = semantic
        #self.pooling = torch.nn.AvgPool2d(kernel_size=2,stride=2)
        self.domain_shift = v2.Compose([
            v2.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.7, hue=0.1),
            #v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            #v2.RandomApply([v2.RandomAutocontrast()], p=0.3),
        ])
    def fixed_domain_shift(self, img):
        img = TF.adjust_brightness(img, 1.2)   # darker
        img = TF.adjust_contrast(img, 1.2)     # higher contrast
        img = TF.adjust_saturation(img, 2)   # less saturated
        img = TF.adjust_hue(img, -0.05)         # slight hue shift
        return img

    def transform(self, im1, im2, lab):

        # Random crop
        i, j, h, w = v2.RandomCrop.get_params(
            im1, output_size=(self.cropsize, self.cropsize))
        im1 = TF.crop(im1, i, j, h, w)
        im2 = TF.crop(im2, i, j, h, w)
        lab = TF.crop(lab, i, j, h, w)

        #im2 = self.fixed_domain_shift(im2)

        # Transform to tensor
        im1 = TF.to_tensor(im1)
        im2 = TF.to_tensor(im2)
        lab = TF.to_tensor(lab)

        return im1, im2, lab

    def compute_semantic_change(self, landcover1, landcover2):
        semantic_change_map = landcover1*10 + landcover2
        return semantic_change_map

    def __len__(self):
        return len(self.im1)
    
    def __getitem__(self, idx):
        image1 = Image.open(self.im1[idx])
        image2 = Image.open(self.im2[idx])
        masks = Image.open(self.lab[idx])
        image1, image2, masks = self.transform(image1, image2, masks)

        #image1 = self.pooling(image1)
        #image2 = self.pooling(image2)

        landcover1 = (masks[0,:,:]*255).unsqueeze(0).unsqueeze(0)
        landcover2 = (masks[1,:,:]*255).unsqueeze(0).unsqueeze(0)
        changemap = (masks[2,:,:]*255).unsqueeze(0).unsqueeze(0)

        #landcover1 = F.interpolate(landcover1.float(), size=(int(self.cropsize/2), int(self.cropsize/2)), mode='nearest').to(torch.int)
        #landcover2 = F.interpolate(landcover2.float(), size=(int(self.cropsize/2), int(self.cropsize/2)), mode='nearest').to(torch.int)
        #changemap = F.interpolate(changemap.float(), size=(int(self.cropsize/2), int(self.cropsize/2)), mode='nearest').to(torch.int)

        semantic_change_map = self.compute_semantic_change(landcover1, landcover2)

        '''# Compute the difference
        if self.type == 'train' or self.type == 'val':
            difference = masks[0,:,:] - masks[1,:,:]
        else:
            difference = masks[1,:,:] - masks[2,:,:]'''

        # Convert to binary change map
        changemap = (changemap == 2).int().unsqueeze(0)

        if self.everything:
            return image1, image2, changemap.squeeze(),changemap.squeeze(), landcover1.long().squeeze(), landcover2.long().squeeze()
        else:
            return image1, landcover1.long().squeeze()
            
label_hierarchy_num_HIUCD = {
    0:{1: {},2:{},3:{},4:{},5:{},6:{},7:{},8:{},9:{}}}
