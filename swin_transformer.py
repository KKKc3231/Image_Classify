# -*- coding:utf-8 -*-
# 作者：KKKC

from timm.models.swin_transformer import SwinTransformer
import torch.nn as nn
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from get_data import Get_Loader

swin_tiny_cfg = dict(patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
swin_tiny = SwinTransformer(**swin_tiny_cfg)

# 若使用预训练好的模型，从swin transformer的github主页下载即可
#https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth

#swin_tiny.load_state_dict(torch.load('C:/Users/HP/Desktop/swin_tiny_patch4_window7_224.pth',map_location=torch.device(device)['model']),strict=True)

# swin 网络架构
class swintrans(nn.Module):
    def __init__(self):
        super(swintrans, self).__init__()
        self.backbone = swin_tiny # swin的总体架构
        #
        self.net = nn.Sequential(
            nn.Linear(768,256),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2), # 两类别
            nn.ReLU(),
            nn.Softmax()
        )
        # self.fc1 = nn.Linear(768,256)
        # self.dp1 = nn.Dropout(0.25)
        # self.fc2 = nn.Linear(256,128)
        # self.relu1 = nn.ReLU()
        # self.dp2 = nn.Dropout(0.2)
        # self.fc3 = nn.Linear(128,2)
        # self.relu2 = nn.ReLU()
        # self.soft = nn.Softmax()

    def forward(self,x):
        x = self.backbone.forward_features(x)
        x = self.net(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir='log/train') # tensorboard
# loader
train_loader, valid_loader = Get_Loader()
print('data load ok.')

net = swintrans().to(device)
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


for epoch in range(1,100):
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
torch.save(net.state_dict(), "model/Swin-trans100.pth")