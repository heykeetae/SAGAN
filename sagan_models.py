import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from spectral import SpectralNorm
import numpy as np

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

class lightSelfAtten(nn.Module):
    ''' channel attention block'''
    def __init__(self, in_channels, activation):
        super(lightSelfAtten, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self,x):
        bh,ch,hi,wh = x.shape
        first = x.view(bh,ch,-1)
        second = x.view(bh,ch,-1).permute(0,2,1)
        third = x.view(bh,ch,-1)
        atten = torch.bmm(first,second)
        atten = torch.max(atten, dim=-1, keepdim=True)[0].expand_as(atten) - atten
        atten = torch.bmm(atten,third).view(bh,ch,hi,wh)
        atten = self.softmax(self.depthwise(atten))
        out = self.gamma*atten + x
        return out, atten



class lightSelfAtten2(nn.Module):
    def __init__(self, in_channels, activation):
        super(lightSelfAtten2,self).__init__()
        
        self.activation = activation
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        bh,ch,hi,wh = x.shape
        x = x.view([bh,ch,hi*wh])
        first = torch.matmul(torch.transpose(x,2,1),x)
        atten = torch.matmul(x,first).view([bh,ch,hi,wh])
        atten = self.depthwise(atten)
        atten = self.pointwise(atten)
        x = x.view([bh,ch,hi,wh])
        out = self.gamma*atten + x 
        # print(self.gamma)
        return out, atten


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


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=64, image_size=64, conv_dim=64):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if self.imsize == 64:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(512, 'relu')

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out,p1 = self.attn1(out)
        out=self.l4(out)
        out,p2 = self.attn2(out)
        out=self.last(out)

        return out.squeeze(), p1, p2


class baseGenBlock(nn.Module):
    """base block for generator"""
    def __init__(self, in_channel=64, out_channel=64, size:int=4, stride=1, padding=0):
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
        self.layer_count = int(np.log2(self.image_size)) - 7
        conv_block = []
        last_layer = []
        curr_shape = self.image_size
        conv_block.append(baseGenBlock(z_dim, self.image_size,4))
        for i in range(2):
            conv_block.append(baseGenBlock(curr_shape, int(curr_shape//2), 4, 2, 1))
            curr_shape = int(curr_shape//2)
        conv_block.append(baseGenBlock(curr_shape, 128,4,2,1))
        self.conv_block = nn.Sequential(*conv_block)
        self.middle_layer = baseGenBlock(128,64,4,2,1)
        curr_shape = 64
        for i in range(self.layer_count):
            last_layer.append(baseGenBlock(curr_shape,int(curr_shape//2), 4, 2, 1))
            curr_shape = int(curr_shape//2)
        last_layer.append(nn.ConvTranspose2d(curr_shape,3,4,2,1))
        last_layer.append(nn.Tanh())
        self.final_layer = nn.Sequential(*last_layer)
        self.attn1 = lightSelfAtten(128,'relu')
        self.attn2 = lightSelfAtten(64,'relu')
        # self.attn1 = Self_Attn( 128, 'relu')
        # self.attn2 = Self_Attn( 64,  'relu')
    
    def forward(self,x):
        # print("layer count in the block : {}".format(self.layer_count))
        # print("input x shape : ", x.size())
        x = x.view(x.size(0), x.size(1), 1, 1)
        # print("input x shape after reshape : ", x.size())
        # out = self.inital_layer(x)
        # print("input out shape after init : ", out.size())
        # out, p1 = self.attn1(out)
        # print("input out shape after atten : ", out.size())
        out = self.conv_block(x)
        # print("input out shape after conv : ", x.size())
        out,p1 = self.attn1(out)
        # print("input out shape after atten : ", out.size())
        out = self.middle_layer(out)
        # print("input out shape after middle : ", out.size())
        out,p2 = self.attn2(out)
        # print("input out shape after atten : ", out.size())
        out = self.final_layer(out)
        # print("input out shape after final : ", out.size())
        return out, p1, p2

class baseEncBlock(nn.Module):
    """base block for encoding"""
    def __init__(self, in_channel=64, out_channel=64, size=4, stride=1, padding=1):
        super(baseEncBlock, self).__init__()
        layer = []
        layer.append(SpectralNorm(nn.Conv2d(in_channel, out_channel, size, stride, padding)))
        layer.append(nn.LeakyReLU(0.1))
        self.layer = nn.Sequential(*layer)
    def forward(self,x):
        out = self.layer(x)
        return out

class hqDiscriminator(nn.Module):
    def __init__(self,batch_size=64, image_size=64, conv_dim=64):
        super(hqDiscriminator,self).__init__()
        conv_block = []
        self.imsize = image_size
        loop_count = int(np.log2(image_size) - 4)
        curr_dim = 2**int(np.log2(256) - loop_count)
        conv_block.append(baseEncBlock(3, curr_dim, 4, 2, 1))
        for i in range(loop_count):
            curr_dim *=2
            conv_block.append(baseEncBlock(curr_dim//2, curr_dim,4,2,1))
        curr_dim *= 2
        self.conv_block = nn.Sequential(*conv_block)
        self.middle_layer = baseEncBlock(curr_dim//2, curr_dim,4,2,1)
        self.final_layer = nn.Conv2d(curr_dim,1,4)
        # self.attn1 = Self_Attn(256, 'relu')
        # self.attn2 = Self_Attn(512, 'relu')
        self.attn1 = lightSelfAtten(256, 'relu')
        self.attn2 = lightSelfAtten(512, 'relu')

    def forward(self,x):
        out = self.conv_block(x)
        out, p1 = self.attn1(out)
        out = self.middle_layer(out)
        out, p2 = self.attn2(out)
        out = self.final_layer(out)
        return out.squeeze(), p1, p2  

class mapEncoder(nn.Module):
    def __init__(self, in_channel, im_size, z_dim):
        super(mapEncoder,self).__init__()
        self.z_dim = z_dim
        self.im_size = im_size
        self.in_channel = in_channel
        self.count = int(np.log2(self.im_size) - 4)
        layer = []
        for i in range(self.count):
            layer.append(baseEncBlock(1,1,4,2,1))
        self.inital = nn.Sequential(*layer)
        self.final = nn.Linear(256,z_dim)
    def forward(self,x):
        out = self.inital(x)
        out = out.view(out.size(0),-1)
        out = self.final(out)
        return out
