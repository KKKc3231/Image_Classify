# 图像分类 🐱 vs 🐕

## 1、说明

以机器学习作业和Kaggle比赛中的入门比赛为练手，自己搭建CNN网络和修改Resnet的全连接层实现简单的猫狗分类的效果。

**额外补充**：Transformer实现图像的分类问题（swin-transformer）

## 2、代码结构

`get_data.py`：构建自己的数据类，放入Dataloader 

`cnn`：搭建自己的CNN网络

`resnet`：修改resnet34后两层，最后一层全连接层的output修改为两类

`show`：简单的界面选择图像送入网络进行训练

<img src="https://gitee.com/kkkcstx/kkkcs/raw/master/img/20220418153522.png" style="zoom:50%;" />

## 3、CNN模型和Resnet

<img src="https://gitee.com/kkkcstx/kkkcs/raw/master/img/model.jpg" style="zoom: 67%;" />

自己搭建的CNN网络架构如上图所示，4个block，一个block分为Conv、Maxpool、Dropout，最后得到的特征图的大小为4x4，通道个数为128，最后展平，两个全连接层-->到二分类。

## 4、模型框架

```python
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
```

## 5、运行

前提是以及安装好torch的包以及tensorboard

`python cnn.py`：对模型进行训练

`show.py`：模型测试，可更改模型

运行结果：

<img src="https://gitee.com/kkkcstx/kkkcs/raw/master/img/20220418174029.png" style="zoom: 20%;" />

<img src="https://gitee.com/kkkcstx/kkkcs/raw/master/img/20220418173908.png" style="zoom:20%;" />

## 6、Later

类别太少了，多分类问题？一类多品种？比如狗好多品种，拉布拉多，金毛，emm之后再看吧，期末了。

