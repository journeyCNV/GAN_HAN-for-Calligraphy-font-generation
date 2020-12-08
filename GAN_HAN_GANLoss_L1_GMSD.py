import os
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets,transforms
from torchvision.utils import save_image
import numpy as np
from torch.utils.data import Dataset 

image_size = 4096  #(64*64)     6*1000*64*64
num_epochs = 200
batch_size = 100
sample_dir = 'HAN_samples'

if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)    #存储路径

# 设备配置   本地使用的是pytorch的CPU版，执行此语句需要GPU版
#torch.cuda.set_device(0) # 这句用来设置pytorch在哪块GPU上运行，这里假设使用序号为0的这块GPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image processing  图像预处理
trans = transforms.Compose([transforms.Normalize([0.5], [0.5])])

#ToTensor将数据分布到(0,1) Normalize将数据分布到[-1,1]

# =======================================================================================

dataset = np.load('dataset.npy')

class MyDataset(Dataset):
    def __init__(self,data):
        self.data = data
    def __len__(self):
        return len(self.data[0])
    def __getitem__(self,idx):
        #data = torch.from_numpy(self.data[idx])
        data_aim = self.data[0][idx]
        data_def = self.data[1][idx]
        aim = trans(data_aim)  #预处理
        defau = trans(data_def)
        return data_aim,data_def

#手动实现ToTensor
dataset_1 = torch.from_numpy(dataset[0]).view(1000,1,64,64).float().div(255)
dataset_4 = torch.from_numpy(dataset[3]).view(1000,1,64,64).float().div(255) #输入相关
    
my_dataset = (dataset_1,dataset_4)
dataset = MyDataset(my_dataset)

#暂时不做标签，shuffle设置为F，保证字可以对上
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size, 
                                          shuffle=True,pin_memory=True)  #GPU可加pin_memory=True加速



class G(nn.Module):
    def __init__(self):
            super().__init__()
            self.fc1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.2))
    
            self.fc2 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=64,kernel_size=2,stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.2))
    
            self.fc3 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.2))
    
            self.fc4 = nn.Sequential(nn.Conv2d(in_channels=128,out_channels=128,kernel_size=2,stride=2),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.2))
    
            self.fc5 = nn.Sequential(nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2))
    
            self.fc6 = nn.Sequential(nn.Conv2d(in_channels=256,out_channels=256,kernel_size=2,stride=2),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2))
    
            self.fc7 = nn.Sequential(nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2))
    
            self.fc8 = nn.Sequential(nn.Conv2d(in_channels=512,out_channels=512,kernel_size=2,stride=2),
            nn.LeakyReLU(0.2)) #nn.BatchNorm2d(num_features=512)
            
            self.fc9 = nn.Sequential(nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=2,stride=2),
                                    nn.BatchNorm2d(num_features=512),
                                    nn.ReLU())
            
            self.fc10 = nn.Sequential(nn.ConvTranspose2d(in_channels=512*2,out_channels=512,kernel_size=3,stride=1),
                                    nn.BatchNorm2d(num_features=512),
                                     nn.ReLU())
            
            self.fc11 = nn.Sequential(nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=3,stride=1),
                                    nn.BatchNorm2d(num_features=256),
                                     nn.ReLU())
            self.fc12 = nn.Sequential(nn.ConvTranspose2d(256,256,2,2,2),
                                      nn.BatchNorm2d(num_features=256),
                                     nn.ReLU())   #----------------------2,2  原本的2,2导致不匹配 换成 222是匹配的
            self.fc13 = nn.Sequential(nn.ConvTranspose2d(in_channels=256*2,out_channels=256,kernel_size=3,stride=1),
                                      nn.BatchNorm2d(num_features=256),
                                     nn.ReLU())
            self.fc14 = nn.Sequential(nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=3,stride=1),
                                      nn.BatchNorm2d(num_features=128),
                                     nn.ReLU())
            self.deconv1 = nn.ConvTranspose2d(128,1,34,2,0)#******猜
            
            self.fc15 = nn.Sequential(nn.ConvTranspose2d(128,128,3,2,2),
                                      nn.BatchNorm2d(num_features=128),
                                     nn.ReLU())#----------------------------------2,2  322匹配
            self.fc16 = nn.Sequential(nn.ConvTranspose2d(in_channels=128*2,out_channels=128,kernel_size=3,stride=1),
                                      nn.BatchNorm2d(num_features=128),
                                     nn.ReLU())
            self.fc17 = nn.Sequential(nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=3,stride=1),
                                      nn.BatchNorm2d(num_features=64),
                                     nn.ReLU())
            self.deconv2 = nn.ConvTranspose2d(64,1,4,2,2)
            
            self.fc18 = nn.Sequential(nn.ConvTranspose2d(64,64,2,2,2),
                                      nn.BatchNorm2d(num_features=64),
                                     nn.ReLU())#-----------------------------2,2
            self.fc19 = nn.Sequential(nn.ConvTranspose2d(in_channels=64*2,out_channels=64,kernel_size=3,stride=1),
                                      nn.BatchNorm2d(num_features=64),
                                     nn.ReLU())
            self.fc20 = nn.Sequential(nn.ConvTranspose2d(64,1,5,1,2),
                                     nn.Tanh())
    def encode(self,x):
        h1 = self.fc1(x)
        h = self.fc2(h1)
        h2 = self.fc3(h)
        h = self.fc4(h2)
        h3 = self.fc5(h)
        h = self.fc6(h3)
        h4 = self.fc7(h)
        return h1,h2,h3,h4
    
    def forward(self,x):
        #h1,h2,h3,h4 = self.encode(x)
     #   cat() received an invalid combination of arguments - got (Tensor, Tensor), but expected one of:
# * (tuple of Tensors tensors, name dim, Tensor out)
# * (tuple of Tensors tensors, int dim, Tensor out)
        H = self.encode(x)
        h = self.fc8(H[3])
        #print(h.shape,H[3].shape) torch.Size([100, 512, 2, 2]) torch.Size([100, 512, 4, 4])
        h = self.fc9(h)
        #print(h.shape) 100 512 4 4 
        h = torch.cat((h,H[3]),1)
        #print(h.shape)  100 1024 4 4
        h = self.fc10(h)
        h = self.fc11(h)
        
        #print(h.shape,H[2].shape)torch.Size([100, 256, 8, 8]) torch.Size([100, 256, 12, 12])
        h = self.fc12(h)
        #print(h.shape) #100 256 12 12
        h = torch.cat((h,H[2]),1)
        
        #h = torch.cat((self.fc12(h),H[2]),1)
        h = self.fc13(h)
        
        h = self.fc14(h)
        T1 = self.deconv1(h)
        h = self.fc15(h)
        h = torch.cat((h,H[1]),1)
        
        
        h = self.fc16(h)
        h = self.fc17(h)
        T2 = self.deconv2(h)
        
        h = torch.cat((self.fc18(h),H[0]),1)
        h = self.fc19(h)
        T3 = self.fc20(h)
        return T1,T2,T3
		


class D(nn.Module):
    def __init__(self):
            super().__init__()
            self.fc1 = nn.Sequential(nn.Conv2d(1,64,4,2,1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU())
            self.Fc1 = nn.Sequential(nn.Conv2d(64,1,32,1,0),nn.Sigmoid())  #卷积核尝试除来的
            
            self.fc2 = nn.Sequential(nn.Conv2d(64,128,4,2,1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU())
            self.Fc2 = nn.Sequential(nn.Conv2d(128,1,4,16,0),nn.Sigmoid())
            
            self.fc3 = nn.Sequential(nn.Conv2d(128,256,4,2,1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU())
            self.Fc3 = nn.Sequential(nn.Conv2d(256,1,8,1,0),nn.Sigmoid())
            
            self.fc4 = nn.Sequential(nn.Conv2d(256,512,4,2,1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU())
            self.Fc4 = nn.Sequential(nn.Conv2d(512,1,4,1,0),nn.Sigmoid())   #全部换成卷积层      完全卷积的神经网络
            
    def forward(self,x):
        x = self.fc1(x)
        D1 = x
        L1 = self.Fc1(D1)
        
        x = self.fc2(x)
        D2=x
        L2 = self.Fc2(D2)
        
        x = self.fc3(x)
        D3=x
        L3 = self.Fc3(D3)
        
        x = self.fc4(x)
        D4=x
        L4 = self.Fc4(D4)
        return L1,L2,L3,L4
		


from torch.nn import functional as F
class GMSDLoss(nn.Module):
    def __init__(self,in_channel=1,mid_channel=None,device = "cuda",criteration = nn.L1Loss):
        super(GMSDLoss,self).__init__()
        if mid_channel == None:
            mid_channel = in_channel
        self.prewitt_x = torch.FloatTensor([[1./3,0,-1./3]]).reshape((1,1,1,3))#.to(device)
        self.prewitt_x = self.prewitt_x.expand((in_channel,mid_channel,3,3))
        self.prewitt_y = self.prewitt_x.transpose(2,3)
        self.avg_filter = torch.FloatTensor([[0.25,0.25],[0.25,0.25]]).reshape((1,1,2,2))#.to(device)
        self.avg_filter = self.avg_filter.expand((in_channel,mid_channel,2,2))
        self.criteration = criteration() # 默认为均方根误差

    def forward(self, src, tar):
        assert src.size() == tar.size()
        src = F.pad(src,(1,0,1,0))
        tar = F.pad(tar,(1,0,1,0))
        avg_src = F.conv2d(src,self.avg_filter)
        avg_tar = F.conv2d(tar,self.avg_filter)
        avg_src = F.pad(avg_src, (1, 0, 1, 0))
        avg_tar = F.pad(avg_tar, (1, 0, 1, 0))
        mr_sq_x = F.conv2d(avg_src,self.prewitt_x,stride=2)
        mr_sq_y = F.conv2d(avg_src,self.prewitt_y,stride=2)
        md_sq_x = F.conv2d(avg_tar,self.prewitt_x,stride=2)
        md_sq_y = F.conv2d(avg_tar,self.prewitt_y,stride=2)
        eps = 1e-7
        # ver 1
        frac1 = mr_sq_x.mul(md_sq_x) + mr_sq_y.mul(md_sq_y)
        frac2 = ((mr_sq_y**2+mr_sq_x**2+eps).sqrt()).mul((md_sq_y**2+md_sq_x**2+eps).sqrt())
        L_lang = (1-torch.abs(frac1/frac2)).mean()
        # ver 2
        beta_mr = mr_sq_x**2+mr_sq_y**2
        beta_md = md_sq_y**2+md_sq_x**2

        theta_mr = torch.atan((mr_sq_y)/(mr_sq_x+eps))
        theta_md = torch.atan((md_sq_y)/(md_sq_x+eps))
        L_D = self.criteration(theta_mr,theta_md)
        gmsd = L_lang
        return gmsd
		

# 把判别器和传输网络迁移到GPU上  用的时候去掉注释
D = D()
G = G()
GMSD = GMSDLoss()

#D = D.to(device)
#G = G.to(device)
#GMAD = GMSD.to(device)

# 定义判别器的损失函数交叉熵及优化器
criterion = nn.BCELoss()   #暂且用简单的交叉熵函数，先不加入参数
criterion1 = nn.L1Loss()  #L1loss
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

#Clamp函数x限制在区间[min, max]内
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()
    
total_step = 10


def CountD(fake_images,labels):
    fake_score1,sum_loss1 = Cal_Loss(D(fake_images[0]),labels)
    fake_score2,sum_loss2 = Cal_Loss(D(fake_images[1]),labels)
    fake_score3,sum_loss3 = Cal_Loss(D(fake_images[2]),labels)
    fake_score = (fake_score1+fake_score2+fake_score3)/3
    sum_loss = (sum_loss1+sum_loss2+sum_loss3)/3
    return fake_score,sum_loss
	

def Cal_Loss(outputs,real_labels): 
    op = outputs[0]
    op = op.view(100,1)
    loss1 = criterion(op,real_labels)
    
    op1 = outputs[1]
    op1 = op1.view(100,1)
    loss2 = criterion(op1,real_labels)
    
    op2 = outputs[2]
    op2 = op2.view(100,1)
    loss3 = criterion(op2,real_labels)
    
    op3 = outputs[3]
    op3 = op3.view(100,1)
    loss4 = criterion(op3,real_labels)
    
    real_score = op3
    sum_loss = loss1+loss2+loss3+loss4
    return real_score,sum_loss
	
	

for epoch in range(num_epochs):
    for i, Images in enumerate(data_loader):
        images,default_images = Images
        
        images = images.to(device)
        images = images.float()  #必须加  
        
        # 定义图像是真或假的标签
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ================================================================== #
        #                      训练鉴别器                                    #
        # ================================================================== #
        #print(images.shape)  100 1 64 64
        # 定义判断器对真图片的损失函数
        outputs = D(images)  #L1 L2 L3 L4   4个Tensor
        real_score,d_loss_real = Cal_Loss(outputs,real_labels)   
        
        # 定义判别器对假图片（即由模板图片生成的图片）的损失函数
        
        default_images = default_images.to(device)
        default_images = default_images.float()  #100 1 64 64
        fake_images = G(default_images)  #生成T1 T2 T3   100 1 64 64 
        
        fake_score,d_loss_fake = CountD(fake_images,fake_labels)
     
        # 得到判别器总的损失函数
        d_loss = d_loss_real + d_loss_fake
        
        # 对生成器、判别器的梯度清零        
        reset_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # ================================================================== #
        #                        训练传输网络                                #
        # ================================================================== #

        # 定义生成器对假图片的损失函数，这里我们要求
        #判别器生成的图片越来越像真图片，故损失函数中
        #的标签改为真图片的标签，即希望生成的假图片，
        #越来越靠近真图片
        #z = torch.randn(batch_size, latent_size).to(device)
        
        fake_images = G(default_images)#T1 T2 T3
        #fake_images = torch.cat((default_images,fake_images),1)  封装了
        f_op_sum,g_loss = CountD(fake_images,real_labels)
        
        # 对生成器、判别器的梯度清零
        #进行反向传播及运行生成器的优化器
        reset_grad()
        #加L1Loss  判断相似度 坐标轴含义明确 对坐标敏感
        loss_g_L1 = criterion1(fake_images[2],images)*100
        
        #加gmsd_loss  计算边缘相似度 梯度赋值相似度
        gmsd_loss = GMSD(images,fake_images[2])
        
        print('****')
        
        g_loss_all = g_loss + loss_g_L1 + gmsd_loss
        g_loss_all.backward()
        g_optimizer.step()
        
        if (i+1) % 5 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                          real_score.mean().item(), fake_score.mean().item()))
    
    # 保存真图片
    if (epoch+1) == 1:
        #images = images.reshape(images.size(0),1,64,64)
        save_image(images, os.path.join(sample_dir, 'real_images.png'))
    
    # 保存假图片
    #fake_images = fake_images[2].reshape(fake_images[2].size(0),1,64,64)    #T3的结果   T1 T2先不看 
    save_image(fake_images[2], os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))

# 保存模型
torch.save(G.state_dict(), 'G.ckpt')
torch.save(D.state_dict(), 'D.ckpt')
