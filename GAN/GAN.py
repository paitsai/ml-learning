
import torch
from torch import optim
from torch.autograd import Variable
import torchvision
import torch.nn as nn
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:#find() 返回字符串第一次出现的索引，如果没有匹配项则返回-1
        m.weight.data.normal_(0.0, 0.02)#归一化
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class G(nn.Module):
    def __init__(self,in_dims=1024,out_dim=512):
        # in_dims指的是输出的随机向量的维度 ; out_dims指的是输出图像的尺寸
        super(G,self).__init__()
        self.model=nn.Sequential()

        self.fn1=nn.Sequential(nn.Linear(in_dims,out_dim*64*64*4),
                               nn.BatchNorm1d(out_dim*64*64*4),
                               nn.ReLU()) #经过第一层线性层之后记得reshape为64*64*1024的形状
        
        self.arcconv1=nn.ConvTranspose2d(in_channels=2048,out_channels=1024,kernel_size=3,stride=1,padding=1,output_padding=0)
        self.rl1=nn.ReLU()
        self.arcconv2=nn.ConvTranspose2d(in_channels=1024,out_channels=256,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.rl2=nn.ReLU()
        self.arcconv3=nn.ConvTranspose2d(in_channels=256,out_channels=64,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.rl3=nn.ReLU()
        self.arcconv4=nn.ConvTranspose2d(in_channels=64,out_channels=3,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.rl4=nn.Tanh()

        self.fn2=nn.Sequential(self.arcconv1,self.rl1,self.arcconv2,self.rl2,self.arcconv3,self.rl3,self.arcconv4,self.rl4)

        self.apply(weights_init)
        

    def forward(self,x):
        z=self.fn1(x)
        z=z.view(-1,2048,64,64)
        z=self.fn2(z)
        return z
    
class D(nn.Module):
    def __init__(self,in_dims=3,dims=64):
        # 输入的数据格式应该是  N*3*512*512
        super(D, self).__init__()
        
        self.model = nn.Sequential()

        def conv_binary_2d(in_dim,out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim,out_dim,5,2,2),
                nn.BatchNorm2d(out_dim),nn.LeakyReLU(0.15)
            )
        self.model.append(nn.Conv2d(in_dims, dims, 10, 8, 1))
        self.model.append(nn.LeakyReLU(0.1))
        self.model.append(conv_binary_2d(dims,2*dims))
        self.model.append(conv_binary_2d(2*dims,4*dims))
        self.model.append(conv_binary_2d(4*dims,8*dims))
        self.model.append(conv_binary_2d(8*dims,16*dims))
        self.model.append(nn.Conv2d(16*dims,1,4))
        self.model.append(nn.Sigmoid())
        self.apply(weights_init)
    
    def forward(self,x):
        # 返回是一个batch-size的一维数组
        z=self.model(x)
        z=z.view(-1)
        return z

# 开始训练！
import os

batch_size=32
feature_dim=128 #设置特征向量的大小

lr=5e-5
n_epoch=100

workspace_dir = '.'
save_dir=os.path.join(workspace_dir,'logs')
os.makedirs(save_dir,exist_ok=True)

G_model = G(in_dims=feature_dim).cuda()
D_model = D(3).cuda()
G_model.train()
D_model.train()

# criterion=nn.BCELoss()
criterion = nn.BCEWithLogitsLoss()
# optimizer
opt_D = torch.optim.Adam(D_model.parameters(), lr=lr, betas=(0.5, 0.999))
opt_G = torch.optim.Adam(G_model.parameters(), lr=lr, betas=(0.5, 0.999))

import random
import numpy as np
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(0)

from torch.utils.data import Dataset, DataLoader
import glob
import os
import torchvision.transforms as transforms
from PIL import Image

import numpy as np



class GAN_dataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        :param image_dir: 存储图像的目录
        :param transform: 可选的转换操作
        """
        
        self.image_dir = image_dir
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)

        return image

#这里可以加载自己想加载的数据
def get_dataset(root):
    t = transforms.ToTensor()
    dataset = GAN_dataset(root, transform=t)
    return dataset

dataset = get_dataset(os.path.join(workspace_dir, 'raw_GAN'))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

import matplotlib.pyplot as plt


# 到这里所有的测试工作就已经全部完成了

# 正式开始训练

for e,epoch in enumerate(range(n_epoch)):
    for i,data in enumerate(dataloader):

        # 第一阶段训练 识别器模型
        img_batch=data
        # print(f"data type={type(data)}")
        # print(img_batch.size())
        img_batch=img_batch.to("cuda")
        
        bsize=img_batch.size(0)
        z_test=Variable(torch.randn(bsize,feature_dim)).cuda()
        generated_imgs=G_model(z_test)
        initial_imgs=Variable(img_batch).cuda()

        generated_labels=torch.zeros(bsize).cuda()
        initial_labels=torch.ones(bsize).cuda()

        generated_predict=D_model(generated_imgs)
        initial_predict=D_model(initial_imgs)

        generated_loss=criterion(generated_labels,generated_predict)
        initial_loss=criterion(initial_labels,initial_predict)

        Dloss=(generated_loss+initial_loss)/2

        D_model.zero_grad()
        Dloss.backward()
        opt_D.step()

        # 第二阶段训练生成器模型

        z_test2=Variable(torch.randn(bsize,feature_dim)).cuda()
        
        generated_imgs2=G_model(z_test2)
        generated_predict=D_model(generated_imgs2)
   
        Gloss=criterion(generated_predict,initial_labels)

        G_model.zero_grad()
        Gloss.backward()
        opt_G.step()

        print(f"epoch :{epoch+1}/{n_epoch},D_model_loss:{Dloss},G_model_loss:{Gloss}",end="\n")

# 训练完了

G_model.eval()
final_generated_imgs=(G_model(z_test)+1)/2
filename = os.path.join(save_dir, f'Epoch_{epoch+1:03d}.jpg')
torchvision.utils.save_image(final_generated_imgs, filename, nrow=10)
print(f' | Save some samples to {filename}.')

   # show generated image图片的可视化
grid_img = torchvision.utils.make_grid(final_generated_imgs.cpu(), nrow=10)
plt.figure(figsize=(10,10))
plt.imshow(grid_img.permute(1, 2, 0))
plt.show()
G.train()
if (e+1) % 5 == 0:
    torch.save(G_model.state_dict(), os.path.join(workspace_dir, f'G_model.pth'))
    torch.save(D_model.state_dict(), os.path.join(workspace_dir, f'D_model.pth'))


       