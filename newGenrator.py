import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from spectral import SpectralNorm
from torch.autograd import Variable



class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention


class baseGenBlock(nn.Module):
    """base block for generator"""
    self.layer
    def __init__(self, in_channel=64, out_channel=64, size=4, stride=1, padding=0):
        super(baseGenBlock,self).__init__()
        layer = []
        layer.append(SpectralNorm(nn.ConvTranspose2d(in_channel, out_channel,size,stride,padding)))
        layer.append(nn.BatchNorm2d(out_channel))
        layer.append(nn.ReLU())
        self.layer = nn.Sequential(*layer)
    
    def forward(self,x):
        out = self.layer(x)
        return out

class hqGenerator(nn.Module):
    '''hqGenerator.'''
    def __init__(self, batch_size, image_size=64, z_dim=128, conv_dim=64):
        super(hqGenerator, self).__init__()
        self.image_size = image_size
        self.batch_size = batch_size
        conv_block = []
        last_layer = []
        out_shape = conv_dim*16
        repeat_num = int(np.log2(self.imsize)) - 3
        conv_block.append(baseGenBlock(z_dim,out_shape,4))
        mult = 2 ** repeat_num 
        while(out_shape>=128):
            conv_block.append(baseGenBlock(out_shape,int(out_shape/2),4, 2,1))
            out_shape = int(out_shape/2)
        self.conv_block = nn.Sequential(*conv_block)
        self.middle_layer = baseGenBlock(128,64,4,2)
        last_layer.append(nn.ConvTranspose2d(64,3,4,2,1))
        last_layer.append(nn.Tanh())
        self.final_layer = nn.Sequential(*last_layer)
        self.attn1 = Self_Attn( 128, 'relu')
        self.attn2 = Self_Attn( 64,  'relu')
    
    def forward(self,x):
        out = self.conv_block(x)
        out,p1 = self.atten1(out)
        out = self.middle_layer(out)
        out,p2 = self.attn2(out)
        self.final_layer(out)
        out = nn.functional.interpolate(out, (self.batch_size, self.image_size,self.image_size))
        return out, p1, p2

class baseDisBlock(nn.Module):
    """base block for discriminator"""
    self.layer
    def __init__(self, in_channel=64, out_channel=64, size=4, stride=1, padding=1):
        super(baseDisBlock, self).__init__()
        layer.append(SpectralNorm(nn.Conv2d(in_channel, out_channel, size, stride)))
        layer.append(nn.LeakyReLU(0.1))
        self.layer = nn.Sequential(*layer)
    def forward(self,x):
        out = self.layer(x)
        return out

class hqDiscriminator(nn.Module):
    def __init__(self,batch_size=64, image_size=64, conv_dim=64):
        super(hqDiscriminator,self).__init__()
        self.imsize = image_size

        conv_block = []
        loop_count = np.log2(self.image_size) - 4
        curr_dim = 2 **int(np.log2(256) - loop_count)
        conv_block.append(baseDisBlock(3, curr_dim,4,2,1))
        for i in range(loop_count):
            curr_dim *=2
            conv_block.append(baseDisBlock(curr_dim//2, curr_dim,4,2,1))
        curr_dim *= 2
        self.conv_block = nn.Sequential(*conv_block)
        self.middle_layer = baseDisBlock(curr_dim//2, curr_dim,4,2,1)
        self.final_layer = nn.Conv2d(curr_dim*2,1,4)
        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(512, 'relu')

    def forward(self,x):
        out = conv_block(x)
        out, p1 = self.attn1(out)
        out = self.middle_layer(out)
        out, p2 = self.attn2(out)
        out = self.final_layer(out)
        return out.squeeze(), p1, p2     

class Generator(nn.Module):
    """Generator."""

    def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64):
        super(Generator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        repeat_num = int(np.log2(self.imsize)) - 3
        mult = 2 ** repeat_num # 8
        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)))
        layer1.append(nn.BatchNorm2d(conv_dim * mult))
        layer1.append(nn.ReLU())

        curr_dim = conv_dim * mult

        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer2.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer3.append(nn.ReLU())

        if self.imsize == 64:
            layer4 = []
            curr_dim = int(curr_dim / 2)
            layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
            layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer4.append(nn.ReLU())
            self.l4 = nn.Sequential(*layer4)
            curr_dim = int(curr_dim / 2)

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn( 128, 'relu')
        self.attn2 = Self_Attn( 64,  'relu')

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out=self.l1(z)
        out=self.l2(out)
        out=self.l3(out)
        out,p1 = self.attn1(out)
        out=self.l4(out)
        out,p2 = self.attn2(out)
        out=self.last(out)

        return out, p1, p2

class Generator(nn.Module):
    """Generator."""

    def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64):
        super(Generator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        repeat_num = int(np.log2(self.imsize)) - 3
        mult = 2 ** repeat_num # 8
        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)))
        layer1.append(nn.BatchNorm2d(conv_dim * mult))
        layer1.append(nn.ReLU())

        curr_dim = conv_dim * mult

        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer2.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer3.append(nn.ReLU())

        if self.imsize == 64:
            layer4 = []
            curr_dim = int(curr_dim / 2)
            layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
            layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer4.append(nn.ReLU())
            self.l4 = nn.Sequential(*layer4)
            curr_dim = int(curr_dim / 2)

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn( 128, 'relu')
        self.attn2 = Self_Attn( 64,  'relu')

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out=self.l1(z)
        out=self.l2(out)
        out=self.l3(out)
        out,p1 = self.attn1(out)
        out=self.l4(out)
        out,p2 = self.attn2(out)
        out=self.last(out)

        return out, p1, p2