# -*- coding:utf-8 -*-
# 作者：KKKC

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

# 定义网络架构
class KKKC(nn.Module):
    def __init__(self):
        super(KKKC, self).__init__()
        # 图片大小为 w x w，filter F x F，步长S, padding为p
        # 则 N = (W-F+2P)/S + 1 ，输出图片的大小为N x N
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.25)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.25)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,96,kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.25)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(96,128,kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.25)
        )
        self.feed = nn.Sequential(
            nn.Linear(128*4*4,128),
            nn.Dropout(0.25),
            nn.Linear(128,2)
        )
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0),-1) # 拉成一个向量
        x = self.feed(x)
        return F.softmax(x,dim=-1)

# 模型加载，load_state_dict只加载模型的参数，需要实例化网络。
net = KKKC()
net.load_state_dict(torch.load('model/cnn_epoch400.pth',map_location=torch.device('cpu')))
# for name,p in net.named_parameters():
#     print('name:{},parameter:{}'.format(name,p))

# 测试图片
image = Image.open('C:/Users/HP/Desktop/Image/cat3.jpg')

# trans
transform_test = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = transform_test(image)
img = img.unsqueeze(0) # 增加一个维度
output = net(img) # 送入网络
print(output)
_,label = torch.max(output,1)
print(label)

# 绘图
plt.imshow(image)
# 1:cat，0:dog
if label==1:
    print('This is a Cat.')
    plt.title('This is a Cat.')
else:
    print('This is a Dog.')
    plt.title('This is a Dog.')

# image.show()
# 不显示x坐标和y坐标
plt.xticks([])
plt.yticks([])
plt.show()



