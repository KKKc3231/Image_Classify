# -*- coding:utf-8 -*-
# 作者：KKKC

import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential
from torchvision import models
from PIL import Image
from torchvision.transforms import transforms
from torch.optim import Adam
import torch.optim as optim
from get_data import Get_Loader # 获取loader
from torch.utils.tensorboard import SummaryWriter # tensorboard

model = models.resnet34(pretrained=True)
modules = list(model.children())[:-2]

print(*modules)
resnet = nn.Sequential(*modules).eval()

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
        return F.softmax(x,dim=-1)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir='log/train') # tensorboard
# loader
train_loader, valid_loader = Get_Loader()

net = Resnet(resnet).to(device)
# 定义损失函数
loss_fn = nn.CrossEntropyLoss()
# opt = Adam(net.parameters(),lr=1e-3)
opt = optim.Adam(net.parameters(),lr=0.001)

# 训练时计算loss和acc
def loss_acc():
    num = 0
    Loss = 0
    correct = 0
    # i是第几个，data包括img和label
    for i,data in enumerate(train_loader):
        img,label = data
        # 将数据放入设备，有GPU用GPU，没有用CPU
        img = img.to(device)
        label = label.to(device)
        output = net(img)
        # max返回最大值的值以及索引，1表示对行进行操作，0表示对列进行操作
        Max,pre_label = torch.max(output,1)
        num = num + label.size(0) # 一轮batch的大小，128
        correct = correct + (pre_label == label).sum().item()
        loss = loss_fn(output,label) # 损失函数
        opt.zero_grad() #
        loss.backward() # 反向传播
        opt.step() # 更新参数
        Loss = Loss + loss.item()
    return Loss/(i+1),correct/num # 返回平均损失和acc

# valid loss 和 acc
def valid_loss_acc():
    num = 0
    Loss = 0
    correct = 0
    for i,data in enumerate(valid_loader):
        img, label = data
        # 将数据放入设备，有GPU用GPU，没有用CPU
        img = img.to(device)
        label = label.to(device)
        output = net(img)
        # max返回最大值的值以及索引，1表示对行进行操作，0表示对列进行操作
        Max, pre_label = torch.max(output, 1)
        num = num + label.size(0)  # 一轮batch的大小，128
        correct = correct + (pre_label == label).sum().item()
    return Loss / (i + 1), correct / num  # 返回平均损失和acc


for epoch in range(1,400):
    net.train()
    train_loss,train_acc = loss_acc()
    writer.add_scalar('loss',float(train_loss),epoch)
    writer.add_scalar('acc',float(train_acc),epoch)

    net.eval()
    valid_loss,valid_acc = valid_loss_acc()
    writer.add_scalar('valid_loss', float(valid_loss), epoch)
    writer.add_scalar('valid_acc', float(valid_acc), epoch)

    if epoch % 10 ==0:
        print("epoch{0}训练集acc:{1}，loss:{2}".format(epoch,train_acc,train_loss))

writer.close()
# 只保存模型的参数
torch.save(net.state_dict(),"Resnet_epoch400.pth")