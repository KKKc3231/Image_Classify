# Swin-transformers实现猫狗识别📚

## 1、说明

- 之前只知道transformer可以应用于NLP任务，像Bert和GPT等模型，了解到transformer也可以应用于解决图像分类、目标检测、语义分割等任务，模型分别对应提出的VIT和Swin transformer。
- 机器学习课程作业为实现图片的分类问题，之前使用的是CNN和Resnet等卷积神经网络，该文档用于记录使用Swin transformer 实现猫狗分类任务

## 2、Swin transformer

---

从transformer到Swin：

- Swin将transformer的注意力机制应用于图像领域，如图片分类、目标检测、语义分割，相比于VIT模型来说，将图像分为更小的patch，然后在每个小patch内各自计算注意力；
- 提出窗口滑动，shift window，使不同的窗口之间能够有通信，起到一定的全局链接；
- 几种特殊的mask，由于shift window的操作，使每个模块的大小不同，采用移位加特殊的mask的方式来确保模块的大小一致，一个模块可能是其他几个patch的组成，一个模块以4x4为例，在一个模块内使用特殊的mask来计算不同各自patch内的有效的注意力（不是一个patch内的不应该计算mask）；
- Swin transformer与VIT的对比图：

<div align=center><img src="https://gitee.com/kkkcstx/kkkcs/raw/master/img/20220502200032.png" style="zoom: 40%;" />

- shift window!

<div align=center><img src="https://gitee.com/kkkcstx/kkkcs/raw/master/img/20220502200443.png" style="zoom: 33%;"> </div>

左图经过移位操作，分为4个部分，4为一个独立的部分，可以直接计算注意力；5和3为一个区域，7和1为一个区域，8，6，2，0为一个区域，对这些区域计算注意力需要使用特殊的mask。

我们希望在计算Attention的时候，**让具有相同index QK进行计算，而忽略不同index QK计算结果**。


<div align=center><img src="https://gitee.com/kkkcstx/kkkcs/raw/master/img/20220502201637.png" style="zoom: 10%;" />



不同位置的mask如上图所示。

## 3、timm

`timm`是pytorch视觉模型库，可通过timm库来调用swin transformer、VIT、resnet等模型。

安装的时候使用`pip`，使用`conda`会自动更新torch的版本，可能会造成你原来torch是cuda版本的，更新后就成cpu版本的了，安装的时候注意别使用conda来安装。

## 4、Code

构建Swin transformer的tiny框架，以其作为backbone网络，并增加几个线性层，最后映射到要分的类别。

网络结构：

```python
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

    def forward(self,x):
        x = self.backbone.forward_features(x)
        x = self.net(x)
        return x
```

具体代码见`swin_transformer.py`

## 5、总结

训练的模型用来实现猫狗分类，一共2000张图片，模型的缺点就是使用官方定义的模型来训练一个简单的分类任务来说，有点大动干戈，模型的参数量太大了，使用swin tiny来作为主干网络来训练需要使用大约48G的显卡内存。由于本人实在没有那么多计算资源，之后条件允许了train一波试试👻。

