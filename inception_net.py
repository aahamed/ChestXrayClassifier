# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 13:04:18 2021

@author: brand
"""

import torch
import torch.nn as nn

n_class = 15

class InceptionBlock(nn.Module):

    def __init__(self, in_ch,out_ch):
        super().__init__()
        self.n_class = n_class
        self.convolve1 = nn.Conv2d(in_ch,out_ch,kernel_size = 1, stride = 1, padding=0)
        self.convolve3 = nn.Conv2d(out_ch,out_ch, kernel_size= 3, stride=1, padding=1)
        self.convolve5 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=2 )
        self.max = nn.MaxPool2d(3,stride=1,padding=1)
        
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self,x):
        y1 = self.bn(self.relu(self.convolve1(x)))
        y13 = self.bn(self.relu(self.convolve3(y1)))
        y15 = self.bn(self.relu(self.convolve5(y1)))
        ymax = self.bn(self.relu(self.convolve1(self.max(x))))
        
        inception = torch.cat([y1,y13], dim=1)
        inception = torch.cat([inception,y15], dim=1)   
        inception = torch.cat([inception,ymax], dim=1) 
        return inception

class BrandNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        
        self.conv1 = nn.Conv2d(3,32, kernel_size = 3, stride = 1, padding = 1)
        self.bnd1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,32, kernel_size = 3, stride = 1, padding = 1)
        self.bnd2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,64,kernel_size = 3, stride = 1, padding = 1)
        self.bnd3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64,64,kernel_size = 3, stride = 1, padding = 1)
        self.bnd4= nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64,128, kernel_size = 3, stride = 1, padding = 1) 
        self.bnd5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128,128, kernel_size = 3, stride = 1, padding = 1)
        self.bnd6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128,256, kernel_size = 3, stride = 1, padding = 1)
        self.bnd7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1)
        self.bnd8 = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256,512, kernel_size = 3, stride =1, padding = 1)
        self.bnd9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512,512, kernel_size = 3, stride =1, padding = 1)
        self.bnd10 = nn.BatchNorm2d(512)
        
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.inception1 = InceptionBlock(32,8)
        self.inception2 = InceptionBlock(64,16)
        self.inception3 = InceptionBlock(128,32)
        self.inception4 = InceptionBlock(256,64)
        
        
        self.upconv = nn.ConvTranspose2d(512,256, 3, stride=1,padding=1)
        self.upconv1 = nn.ConvTranspose2d(256,128, 3, stride=1,padding=1)
        self.upconv2 = nn.ConvTranspose2d(128,64, 3, stride=1,padding=1)
        self.upconv3 = nn.ConvTranspose2d(64,32, 3, stride=1,padding=1)
        self.expand = nn.Upsample(scale_factor = 2, mode='nearest')
        self.down = nn.MaxPool2d(2, stride=2)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        
        
        
        self.conv11 = nn.Conv2d(512,256, kernel_size = 3, stride =1, padding = 1)
        self.bnd11 = nn.BatchNorm2d(256)
        self.conv12 = nn.Conv2d(256,256,kernel_size = 3, stride =1, padding = 1)
        self.bnd12 = nn.BatchNorm2d(256)
        self.conv13 = nn.Conv2d(256,128, kernel_size = 3, stride =1, padding = 1)
        self.bnd13 = nn.BatchNorm2d(128)
        self.conv14 = nn.Conv2d(128,128,kernel_size = 3, stride =1, padding = 1)
        self.bnd14 = nn.BatchNorm2d(128)
        self.conv15 = nn.Conv2d(128,64,kernel_size = 3, stride =1, padding = 1)
        self.bnd15 = nn.BatchNorm2d(64)
        self.conv16 = nn.Conv2d(64,64,kernel_size = 3, stride =1, padding = 1)
        self.bnd16 = nn.BatchNorm2d(64)
        self.conv17 = nn.Conv2d(64,32, kernel_size = 3, stride =1, padding = 1)
        self.bnd17 = nn.BatchNorm2d(32)
        self.conv18 = nn.Conv2d(32,32, kernel_size = 3, stride =1, padding = 1)
        self.bnd18 = nn.BatchNorm2d(32)
        
    def forward(self, x):
        #forward pass encoding 
        
        x1 = self.bnd1(self.relu(self.conv1(x)))
        
        x2 = self.inception1(x1) #use for cat/new inception 
    
        x3 = self.down(x2)    
        x4 = self.bnd3(self.relu(self.conv3(x3)))
        x5 = self.inception2(x4)#cat
       
        x6 = self.down(x5)
      
        x7 = self.bnd5(self.relu(self.conv5(x6)))
        x8 = self.inception3(x7)#cat
        x9 = self.down(x8)
        x10 = self.bnd7(self.relu(self.conv7(x9))) 
        x11 = self.inception4(x10)#cat
        x12 = self.down(x11)
        x13 = self.bnd9(self.relu(self.conv9(x12)))
        x14 = self.bnd10(self.relu(self.conv10(x13)))
        
        #expansion
        
        first = self.relu(self.upconv(x14))
        first = self.expand(first)
        second = torch.cat([x11,first], dim=1)
        
        x15 = self.bnd11(self.relu(self.conv11(second)))
        x16 = self.inception4(x15)#here
        
        third = self.relu(self.upconv1(x16))
        third = self.expand(third)
        fourth = torch.cat([x8,third], dim=1)
        
        x17 = self.bnd13(self.relu(self.conv13(fourth)))
        x18 = self.inception3(x17)#here
        
        fifth = self.relu(self.upconv2(x18))
        fifth = self.expand(fifth)
        sixth = torch.cat([x5,fifth], dim=1)
        
        x19 = self.bnd15(self.relu(self.conv15(sixth)))
        x20 = self.inception2(x19)#here
        
        seventh = self.relu(self.upconv3(x20))
        seventh = self.expand(seventh)
        eighth = torch.cat([x2,seventh], dim=1)
        
        x21 = self.bnd17(self.relu(self.conv17(eighth)))
        x22 = self.inception1(x21)#here
        
        out_decoder = x22
        score = self.classifier(out_decoder)
     
        
        return score