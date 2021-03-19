# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 12:52:37 2021

@author: brand
"""

import torch
import torch.nn as nn

class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        
        self.conv1 = nn.Conv2d(3,64, kernel_size = 3, stride = 1, padding = 1)
        self.bnd1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,64, kernel_size = 3, stride = 1, padding = 1)
        self.bnd2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128,kernel_size = 3, stride = 1, padding = 1)
        self.bnd3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,128,kernel_size = 3, stride = 1, padding = 1)
        self.bnd4= nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128,256, kernel_size = 3, stride = 1, padding = 1) 
        self.bnd5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1)
        self.bnd6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256,512, kernel_size = 3, stride = 1, padding = 1)
        self.bnd7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512,512, kernel_size = 3, stride = 1, padding = 1)
        self.bnd8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512,1024, kernel_size = 3, stride =1, padding = 1)
        self.bnd9 = nn.BatchNorm2d(1024)
        self.conv10 = nn.Conv2d(1024,1024, kernel_size = 3, stride =1, padding = 1)
        self.bnd10 = nn.BatchNorm2d(1024)
        
        self.classifier = nn.Conv2d(64, self.n_class, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.upconv = nn.ConvTranspose2d(1024,512, 3, stride=1,padding=1)
        self.upconv1 = nn.ConvTranspose2d(512,256, 3, stride=1,padding=1)
        self.upconv2 = nn.ConvTranspose2d(256,128, 3, stride=1,padding=1)
        self.upconv3 = nn.ConvTranspose2d(128,64, 3, stride=1,padding=1)
        self.expand = nn.Upsample(scale_factor = 2, mode='nearest')
        self.down = nn.MaxPool2d(2, stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        
        
        
        self.conv11 = nn.Conv2d(1024,512, kernel_size = 3, stride =1, padding = 1)
        self.bnd11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512,512,kernel_size = 3, stride =1, padding = 1)
        self.bnd12 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512,256, kernel_size = 3, stride =1, padding = 1)
        self.bnd13 = nn.BatchNorm2d(256)
        self.conv14 = nn.Conv2d(256,256,kernel_size = 3, stride =1, padding = 1)
        self.bnd14 = nn.BatchNorm2d(256)
        self.conv15 = nn.Conv2d(256,128,kernel_size = 3, stride =1, padding = 1)
        self.bnd15 = nn.BatchNorm2d(128)
        self.conv16 = nn.Conv2d(128,128,kernel_size = 3, stride =1, padding = 1)
        self.bnd16 = nn.BatchNorm2d(128)
        self.conv17 = nn.Conv2d(128,64, kernel_size = 3, stride =1, padding = 1)
        self.bnd17 = nn.BatchNorm2d(64)
        self.conv18 = nn.Conv2d(64,64, kernel_size = 3, stride =1, padding = 1)
        self.bnd18 = nn.BatchNorm2d(64)
        
    def forward(self, x):
        #forward pass encoding 
        
        x1 = self.bnd1(self.relu(self.conv1(x)))
        
        x2 = self.bnd2(self.relu(self.conv2(x1)))#use for cat
        x3 = self.down(x2)    
        x4 = self.bnd3(self.relu(self.conv3(x3)))
        x5 = self.bnd4(self.relu(self.conv4(x4)))#cat
       
        x6 = self.down(x5)
      
        x7 = self.bnd5(self.relu(self.conv5(x6)))
        x8 = self.bnd6(self.relu(self.conv6(x7)))#cat
        x9 = self.down(x8)
        x10 = self.bnd7(self.relu(self.conv7(x9))) 
        x11 = self.bnd8(self.relu(self.conv8(x10)))#cat
        x12 = self.down(x11)
        x13 = self.bnd9(self.relu(self.conv9(x12)))
        x14 = self.bnd10(self.relu(self.conv10(x13)))
        
        #decoder
        
        first = self.upconv(x14)
        first = self.expand(first)
        second = torch.cat([x11,first], dim=1)
        
        x15 = self.bnd11(self.relu(self.conv11(second)))
        x16 = self.bnd12(self.relu(self.conv12(x15)))
        
        third = self.upconv1(x16)
        third = self.expand(third)
        fourth = torch.cat([x8,third], dim=1)
        
        x17 = self.bnd13(self.relu(self.conv13(fourth)))
        x18 = self.bnd14(self.relu(self.conv14(x17)))
        
        fifth = self.upconv2(x18)
        fifth = self.expand(fifth)
        sixth = torch.cat([x5,fifth], dim=1)
        
        x19 = self.bnd15(self.relu(self.conv15(sixth)))
        x20 = self.bnd16(self.relu(self.conv16(x19)))
        
        seventh = self.upconv3(x20)
        seventh = self.expand(seventh)
        eighth = torch.cat([x2,seventh], dim=1)
        
        x21 = self.bnd17(self.relu(self.conv17(eighth)))
        x22 = self.bnd18(self.relu(self.conv18(x21)))
        
        out_decoder = x22
        score = self.classifier(out_decoder)
     
        
        return score