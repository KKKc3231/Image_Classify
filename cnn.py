# -*- coding:utf-8 -*-
# 作者：KKKC

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from get_data import Get_Loader # 获取loader
from torch.utils.tensorboard import SummaryWriter # tensorboard

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
        return F.log_softmax(x,dim=-1)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir='log/train') # tensorboard
# loader
train_loader, valid_loader = Get_Loader()

net = KKKC().to(device)
# 定义损失函数
loss_fn = nn.CrossEntropyLoss()
# 优化器
opt = optim.Adam(net.parameters(),lr=1e-3)

# 训练时计算loss和acc
def loss_acc():
    num = 0
    Loss = 0
    correct = 0
    # i是第几个，data包括img和label
    for i,data in enumerate(train_loader):
        img,label = data
        # 将数据放入设备
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
    return Loss/(i+1),correct/num

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
    return Loss / (i + 1), correct / num


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
torch.save(net.state_dict(), "model/cnn_epoch401.pth")



