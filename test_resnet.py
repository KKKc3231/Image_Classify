# -*- coding:utf-8 -*-
# 作者：KKKC
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

# resnet
class Resnet(nn.Module):
    def __init__(self,resnet):
        super(Resnet, self).__init__()
        self.resnet = resnet
        self.feed = nn.Sequential(
            nn.Linear(512 * 2 * 2, 256),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )
    def forward(self,x):
        x = self.resnet(x)
        x = x.view(x.size(0),-1) # 展平
        x = self.feed(x)
        return F.log_softmax(x,dim=-1)

# 测试图片
image = Image.open('C:/Users/HP/Desktop/Image/cat.jpg')
# trans
transform_test = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = transform_test(image)
img = img.unsqueeze(0) # 增加一个维度

model = models.resnet34(pretrained=True)
modules = list(model.children())[:-2]
resnet = nn.Sequential(*modules).eval()
res = Resnet(resnet)
res.load_state_dict(torch.load('model/Resnet_epoch100.pth',map_location=torch.device('cpu')))

output = res(img)
_,label = torch.max(output,dim=1)

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