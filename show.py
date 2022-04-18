import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import tkinter
import tkinter.filedialog
from PIL import Image,ImageTk
from torchvision import transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

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

#创建一个界面窗口
win = tkinter.Tk()
win.title("Dog_Cat")
win.geometry("700x400")

# trans_test
transform_test = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 选择图片
def choose_img():
    select_file = tkinter.filedialog.askopenfilename(title='选择图片')
    load = Image.open(select_file)

    # 用来显示的图片
    load = transforms.Resize((300,320))(load)
    # 加载模型
    net = KKKC()
    net.load_state_dict(torch.load(r'G:\Dogs_Cats\model\cnn_epoch400.pth', map_location=torch.device('cpu')))

    # 将图片转化为测试的size
    img_test = transform_test(load)
    img2 = img_test.unsqueeze(0)  # 增加一个维度
    output = net(img2)  # 送入网络
    _, label = torch.max(output, 1)
    print(label)
    # 1:cat，0:dog
    if label == 1:
        label1 = tkinter.Label(win, font = ('Times New Roman',20),text="This is a cat!")
        label1.place(x=450, y=250)
    else:
        label2 = tkinter.Label(win, font = ('Times New Roman',20),text="This is a dog!")
        label2.place(x=450, y=250)

    render = ImageTk.PhotoImage(load)
    img = tkinter.Label(win, image=render)
    img.image = render
    img.place(x=50, y=50)

#设置选择图片的按钮
button1 = tkinter.Button(win, text ="选择图片", command = choose_img)
button1.place(x=490,y=100)
win.mainloop()

