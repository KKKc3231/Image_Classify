# -*- coding:utf-8 -*-
# 作者：KKKC

import os
import torch
import torchvision
from PIL import Image
from torchvision import datasets,transforms
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split

def Get_Loader():
    #
    root_dir = 'data/train'
    img_name = os.listdir(root_dir)

    # 用来存储图片位置及标签的数组
    imgs_path = []
    labels_data = []

    # 将dog，cat转化为标签0，1
    for name in img_name:
        if name[:3] == "dog":
            label = 0
        if name[:3] == "cat":
            label = 1
        img_path = os.path.join(root_dir,name)
        imgs_path.append(img_path)
        labels_data.append(label)

    # 划分训练集和测试集，8:2
    train_imgs_path,valid_imgs_path,train_labels,valid_labels = train_test_split(imgs_path,labels_data,test_size=0.2,shuffle=True)

    # 定义transform
    transform_train = transforms.Compose([
        transforms.Resize(70),
        transforms.RandomResizedCrop(64),
        transforms.ToTensor(),
        # ImageNet的参数
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_valid = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 自定义自己的读取数据的类
    class Mydataset(Dataset):
        def __init__(self,imgs_path,labels,transform):
            self.imgs_path = imgs_path
            self.label = labels
            self.transform = transform
            self.len = len(self.imgs_path)

        def __getitem__(self, id):
            img = Image.open(self.imgs_path[id])
            return self.transform(img),self.label[id]

        def __len__(self):
            return self.len

    train_data = Mydataset(train_imgs_path,train_labels,transform_train)
    valid_data = Mydataset(valid_imgs_path,valid_labels,transform_valid)

    Loader_train = DataLoader(train_data,batch_size=128,shuffle=True)
    Loader_valid = DataLoader(valid_data,batch_size=128)
    return Loader_train,Loader_valid




