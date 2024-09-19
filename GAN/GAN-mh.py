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
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # 残差连接
        out = self.relu(out)
        return out



class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # 残差连接
        return out

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, x):
        # x shape: [batch_size, channels, height, width], need to reshape to [height * width, batch_size, channels]
        batch_size, channels, height, width = x.size()
        # print(x.size())
        x = x.view(batch_size, channels, -1).permute(2, 0, 1)  # [height * width, batch_size, channels]
        
        attn_output, _ = self.multihead_attn(x, x, x)  # Self-attention
        attn_output = attn_output.permute(1, 2, 0).view(batch_size, channels, height, width)  # Restore original shape
        return attn_output

class G(nn.Module):
    def __init__(self, in_dims=128):
        super(G, self).__init__()

        self.fn1 = nn.Sequential(
            nn.Linear(in_dims, 512 * 16*4),
            nn.BatchNorm1d(512 * 16*4),
            nn.ReLU()
        )

        self.initial_conv = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
        self.BatchNorm2d1 = nn.BatchNorm2d(128)

        self.second_conv = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.BatchNorm2d2 = nn.BatchNorm2d(64)

        self.attention = MultiHeadAttentionLayer(embed_dim=64, num_heads=4)
        
        self.third_conv = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.BatchNorm2d3 = nn.BatchNorm2d(32)

        self.fourth_conv = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.BatchNorm2d4 = nn.BatchNorm2d(32)

        # 多头自注意力层
        

        # 增加更多残差块
        self.resblocks = nn.Sequential(
            ResBlock(32),
            ResBlock(32),
            ResBlock(32)
        )

        self.fifth_conv = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.BatchNorm2d5 = nn.BatchNorm2d(16)

        self.final_conv = nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

        self.apply(weights_init)

    def forward(self, x):
        z = self.fn1(x)
        z = z.view(-1, 128*4, 8, 8)  # 调整视图以匹配初始卷积层的输入
        
        z = self.initial_conv(z)
        z = self.BatchNorm2d1(z)
        z = nn.ReLU()(z)

        z = self.second_conv(z)
        z = self.BatchNorm2d2(z)
        z = nn.ReLU()(z)
        z = self.attention(z)

        z = self.third_conv(z)
        z = self.BatchNorm2d3(z)
        z = nn.ReLU()(z)

        z = self.fourth_conv(z)
        z = self.BatchNorm2d4(z)
        z = nn.ReLU()(z)

        # 应用多头自注意力
        # z = self.attention(z)

        # 使用多个残差块
        z = self.resblocks(z)

        z = self.fifth_conv(z)
        z = self.BatchNorm2d5(z)
        z = nn.ReLU()(z)

        z = self.final_conv(z)
        return self.tanh(z)

 
class ResBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 确保输入和输出通道匹配
        self.match_dimensions = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.match_dimensions is not None:
            identity = self.match_dimensions(identity)
        
        out += identity  # 残差连接
        out = self.relu(out)
        return out

class D(nn.Module):
    def __init__(self, in_dims=3, dims=32):
        super(D, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_dims, dims, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(dims),
            self.conv_binary_2d(dims, 4 * dims),
            # ResBlock2(2 * dims, 4 * dims),
            self.conv_binary_2d(4 * dims, 4 * dims),
            ResBlock2(4 * dims, 4 * dims),
            self.conv_binary_2d(4 * dims, 8 * dims),
            #ResBlock2(4 * dims, 8 * dims),
            self.conv_binary_2d(8 * dims, 8 * dims),
            #ResBlock2(8 * dims, 8 * dims),
            self.conv_binary_2d(8 * dims, 4 * dims),
            self.conv_binary_2d(4 * dims, dims),
            ResBlock2(dims, dims),
            nn.Conv2d(dims, 1, kernel_size=2*2)
        )

        self.sigm = nn.Sigmoid()
        self.apply(weights_init)

    def conv_binary_2d(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        z = self.model(x)
        z = z.view(-1)  # 确保这与预期的形状匹配
        z = self.sigm(z)
        return z
    
    
# 模型参数设置
batch_size=64
feature_dim=128 #设置特征向量的大小


lr=6e-4
n_epoch=50

# lr=5e-5
# n_epoch=50

device='cuda'
step_ratio=3

# lr=0.000055
# n_epoch=3000

workspace_dir = '.'
save_dir=os.path.join(workspace_dir,'logs')
os.makedirs(save_dir,exist_ok=True)

G_model = G(in_dims=feature_dim).cuda()
D_model = D(3).cuda()
G_model.train()
D_model.train()


criterion=nn.BCELoss().to('cuda')



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
    transforms.Resize((64*8,64*8)),
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


# 优化器定义
opt_D = optim.AdamW(D_model.parameters(), lr=lr, betas=(0.7, 0.999),weight_decay=0.03)
opt_G = optim.AdamW(G_model.parameters(), lr=lr, betas=(0.7, 0.999),weight_decay=0.03)
# optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# 添加余弦学习率调度器
scheduler_D = optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=n_epoch)
scheduler_G = optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=n_epoch)

for epoch in range(n_epoch):
    for idx, real_imgs in enumerate(dataloader):
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)

        if (idx + 6) % 10 == 0 or (idx+9)%10==0:
            real_labels = torch.ones(batch_size).to(device) - 0.0
            fake_labels = torch.zeros(batch_size).to(device) + 0.0
        else:
            real_labels = torch.ones(batch_size).to(device) - 0.08
            fake_labels = torch.zeros(batch_size).to(device) + 0.08

        for j in range(step_ratio):
            z = torch.randn(batch_size, feature_dim).to(device)
            fake_images = G_model(z)
            opt_G.zero_grad()
            g_loss = criterion(torch.squeeze(D_model(fake_images)), real_labels)
            g_loss.backward()
            opt_G.step()

        # 生成虚假图像
        z = torch.randn(batch_size, feature_dim).to(device)
        fake_images = G_model(z)

        opt_D.zero_grad()
        real_loss = criterion(torch.squeeze(D_model(real_imgs)), real_labels)
        fake_loss = criterion(torch.squeeze(D_model(fake_images.detach())), fake_labels)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        opt_D.step()

    # 更新学习率
    
        
        print(f"Epoch [{epoch+1}/{n_epoch}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")
        # if (epoch + 1) % 2 == 0:
        if True:
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
                torch.save(G_model.state_dict(), 'generator_tmp.pth')
                torch.save(D_model.state_dict(), 'discriminator_tmp.pth')
                
    scheduler_D.step()
    scheduler_G.step()
    
    if True:
        torch.save(G_model.state_dict(), 'generator'+f'_{epoch}'+'.pth')
        torch.save(D_model.state_dict(), 'discriminator'+f'_{epoch}'+'.pth')
        torch.save(G_model.state_dict(), 'generator.pth')
        torch.save(D_model.state_dict(), 'discriminator.pth')

# Save the trained model
torch.save(G_model.state_dict(), 'generator.pth')
torch.save(D_model.state_dict(), 'discriminator.pth')
        
