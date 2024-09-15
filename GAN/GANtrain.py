# 导入依赖
import torch
from torch import optim
from torch.autograd import Variable
import torchvision
import torch.nn as nn
import os
import matplotlib.pyplot as plt



os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# 定义初始化方法

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:  # 对卷积层进行 He 初始化
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
    elif classname.find('BatchNorm') != -1:  # 对 BatchNorm 层
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:  # 对全连接层
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')


class ResBlock(nn.Module):
    # 图片的尺度不变，只是改变了通道数量
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.22,inplace=True)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入和输出通道不同，使用 1x1 卷积调整维度
        if in_channels != out_channels:
            self.shortcut = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = None
            
        self.apply(weights_init)

    def forward(self, x):
        identity = x  # 保存输入以进行跳跃连接
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        out += identity  # 加入跳跃连接
        out = self.relu(out)
        return out
    
class G(nn.Module):
    def __init__(self, in_dims=1024, out_dim=512):
        super(G, self).__init__()
        self.model = nn.Sequential()

        self.fn1 = nn.Sequential(
            nn.Linear(in_dims, out_dim * 32),
            nn.BatchNorm1d(out_dim * 32),
            nn.ReLU()
        )

        # 第一层线性后 reshape
        self.initial_conv = nn.ConvTranspose2d(in_channels=512*2, out_channels=256*2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.BatchNorm2d1=nn.BatchNorm2d(512)
        self.second_conv = nn.ConvTranspose2d(256*2, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.BatchNorm2d2=nn.BatchNorm2d(64)
        self.third_conv = nn.ConvTranspose2d(64, 48, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.BatchNorm2d3=nn.BatchNorm2d(48)
        self.resblock2 = ResBlock(48, 8)        
        self.final_conv = nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tanh = nn.Tanh()
        self.apply(weights_init)

    def forward(self, x):
        z = self.fn1(x)
        z = z.view(-1, 512*2, 4, 4)
        z = self.initial_conv(z)
        z=self.BatchNorm2d1(z)
        z=self.second_conv(z)
        z=self.BatchNorm2d2(z)

        # 通过残差块
        # z = self.resblock1(z)
        z=self.third_conv(z)
        z=self.BatchNorm2d3(z)
        
        z = self.resblock2(z)

        z = self.final_conv(z)
        return self.tanh(z)
    
class ResBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2,inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入和输出通道不同，使用 1x1 卷积调整维度
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = None
        self.apply(weights_init)

    def forward(self, x):
        identity = x  # 保存输入以进行跳跃连接
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        out += identity  # 加入跳跃连接
        out = self.relu(out)
        return out

class D(nn.Module):
    def __init__(self, in_dims=3, dims=32):
        super(D, self).__init__()

        self.model = nn.Sequential()

        def conv_binary_2d(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.15)
            )
        
        # 初始卷积层
        # self.model.append(nn.Conv2d(in_dims, dims, kernel_size=10, stride=8, padding=1))
        self.model.append(nn.Conv2d(in_dims, dims, kernel_size=3, stride=2, padding=1))  # 第一个卷积
        nn.BatchNorm2d(dims),
        self.model.append(nn.LeakyReLU(0.12))
        
      
        self.model.append(conv_binary_2d(dims, 2 * dims))
        nn.BatchNorm2d(2*dims)
        self.model.append(ResBlock2(2 * dims, 4 * dims))  # 添加残差块
        self.model.append(conv_binary_2d(4 * dims, 4 * dims))
        nn.BatchNorm2d(4*dims)
        self.model.append(ResBlock2(4 * dims, 2 * dims))  # 添加残差块
        self.model.append(conv_binary_2d(2 * dims, 1 * dims))
        nn.BatchNorm2d(1*dims)
        nn.LeakyReLU(0.2,inplace=True)
        # 输出层
        self.model.append(nn.Conv2d(1 * dims, 1, kernel_size=4))
        
        self.sigm=nn.Sigmoid()
        self.apply(weights_init)

    def forward(self, x):
        z = self.model(x)
        z = z.view(-1)
        z=self.sigm(z)
        
        return z
    
    
# 模型参数设置
batch_size=512
feature_dim=128 #设置特征向量的大小

lr=0.0002
n_epoch=100

workspace_dir = '.'
save_dir=os.path.join(workspace_dir,'logs')
os.makedirs(save_dir,exist_ok=True)

G_model = G(in_dims=feature_dim).cuda()
D_model = D(3).cuda()
G_model.train()
D_model.train()


criterion=nn.BCELoss().to('cuda')
opt_D = torch.optim.Adam(D_model.parameters(), lr=lr, betas=(0.5, 0.999))
opt_G = torch.optim.Adam(G_model.parameters(), lr=lr, betas=(0.5, 0.999))



# 数据准备
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
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.png')]
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
from torchvision import transforms

def get_dataset(root):
    # 定义转换，包括调整大小和转换为张量
    t = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Changing the pixel values in between -1 to 1 
])
    dataset = GAN_dataset(root, transform=t)
    return dataset

dataset = get_dataset(os.path.join(workspace_dir, 'raw_GAN'))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)



if os.path.exists('generator.pth') and os.path.exists('discriminator.pth'):
    # Load the models
    G_model.load_state_dict(torch.load('generator.pth'))
    D_model.load_state_dict(torch.load('discriminator.pth'))
    print("Models loaded successfully.")
else:
    print("No saved models found. Starting training from scratch.")



device='cuda'

step_ratio=2

for epoch in range(n_epoch):
    for i,real_imgs in enumerate(dataloader):
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)
        real_labels = torch.ones(batch_size).to(device)-0.1
        fake_labels = torch.zeros(batch_size).to(device)+0.1
        for i in range(step_ratio):
            
            z = torch.randn(batch_size, feature_dim).to(device)
            fake_images = G_model(z)
            opt_G.zero_grad()
            g_loss=criterion(torch.squeeze(D_model(fake_images)),real_labels)
            g_loss.backward()
            opt_G.step()

        # Ground truths
        z = torch.randn(batch_size, feature_dim).to(device)
        fake_images = G_model(z)
            
            
        opt_D.zero_grad()
        real_loss=criterion(torch.squeeze(D_model(real_imgs)),real_labels)
        fake_loss=criterion(torch.squeeze(D_model(fake_images.detach())),fake_labels)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        opt_D.step()
        

        
        print(f"Epoch [{epoch+1}/{n_epoch}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")
        if (epoch + 1) % 2 == 0:
            with torch.no_grad():
                z = torch.randn(16, feature_dim).to(device)
                fake_images = G_model(z).cpu()
                fake_images = (fake_images + 1) / 2  # Denormalize
                
                fig, axs = plt.subplots(4, 4, figsize=(10, 10))
                for i in range(4):
                    for j in range(4):
                        axs[i, j].imshow(fake_images[i*4 + j].permute(1, 2, 0))
                        axs[i, j].axis('off')
                plt.tight_layout()
                plt.savefig(f'./logs/GAN_epoch_{epoch+1}.png')
                plt.close()
            if (epoch+1)%10==0:
                torch.save(G_model.state_dict(), 'generator.pth')
                torch.save(D_model.state_dict(), 'discriminator.pth')

# Save the trained model
torch.save(G_model.state_dict(), 'generator.pth')
torch.save(D_model.state_dict(), 'discriminator.pth')
        
