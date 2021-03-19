# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 00:00:42 2021

@author: brand
"""

class ResNet50( nn.Module ):
    
    def __init__( self, mode ):
        super().__init__()
        resnet = models.resnet50( pretrained=True )
        self.out_ch = 2048
        
        if mode == 'feature-extract':
            for param in resnet.parameters():
                param.requires_grad = False
        
        # remove fc, avgpool layer from resnet
        self.encoder = nn.Sequential()
        for name, module in resnet.named_children():
            if name == 'fc' or name == 'avgpool':
                continue
            self.encoder.add_module( name, module )

    def forward( self, x ):
        return self.encoder( x )

class UNet(nn.Module):

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
        
        
        
        self.relu = nn.ReLU(inplace=True)
        
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
        
        
     
        
        return x22

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
        
        self.relu = nn.ReLU(inplace=True)
        
        self.inception = InceptionBlock(3,8)
        self.inception1 = InceptionBlock(32,8)
        self.inception2 = InceptionBlock(64,16)
        self.inception3 = InceptionBlock(128,32)
        self.inception4 = InceptionBlock(256,64)
        self.inceptionconv3 = InceptionBlock(32,16)
        self.incetpionconv5 = InceptionBlock(64,32)
        self.inceptionconv7 = InceptionBlock(128,64)
        self.inceptionconv9 = InceptionBlock(256,128)
        self.inceptionfinal = InceptionBlock(512,128)
        self.inceptionup1 = InceptionBlock(512,64)
        self.inceptionup2 = InceptionBlock(256,32)
        self.inceptionup3 = InceptionBlock(128,16)
        self.inceptionup4 = InceptionBlock(64,8)
        
        
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
        
        
    def forward(self, x):
        #forward pass encoding 
        
        x1 = self.inception(x)
        
        x2 = self.inception1(x1) #use for cat/new inception 
    
        x3 = self.down(x2)    
        x4 = self.inceptionconv3(x3)
        x5 = self.inception2(x4)#cat
       
        x6 = self.down(x5)
      
        x7 = self.incetpionconv5(x6)
        x8 = self.inception3(x7)#cat
        x9 = self.down(x8)
        x10 = self.inceptionconv7(x9)
        x11 = self.inception4(x10)#cat
        x12 = self.down(x11)
        x13 = self.inceptionconv9(x12)
        x14 = self.inceptionfinal(x13)
        
        #expansion
        
        first = self.relu(self.upconv(x14))
        first = self.expand(first)
        second = torch.cat([x11,first], dim=1)
        
        x15 = self.inceptionup1(second)
        x16 = self.inception4(x15)#here
        
        third = self.relu(self.upconv1(x16))
        third = self.expand(third)
        fourth = torch.cat([x8,third], dim=1)
        
        x17 = self.inceptionup2(fourth)
        x18 = self.inception3(x17)#here
        
        fifth = self.relu(self.upconv2(x18))
        fifth = self.expand(fifth)
        sixth = torch.cat([x5,fifth], dim=1)
        
        x19 = self.inceptionup3(sixth)
        x20 = self.inception2(x19)#here
        
        seventh = self.relu(self.upconv3(x20))
        seventh = self.expand(seventh)
        eighth = torch.cat([x2,seventh], dim=1)
        
        x21 = self.inceptionup4(eighth)
        x22 = self.inception1(x21)#here
        
        
        return x22
    
class InceptionBlockNew(nn.Module):

    def __init__(self, in_ch,scale):
        super().__init__()
        self.n_class = n_class
        self.convolve1 = nn.Conv2d(in_ch,32*scale,kernel_size = 1, stride = 1, padding=0)
        self.convolve11 = nn.Conv2d(in_ch,128*scale,kernel_size=1, stride=1, padding=0)
        self.convolve111 = nn.Conv2d(in_ch,64*scale,kernel_size=1,stride=1, padding=0)
        self.convolve3 = nn.Conv2d(128*scale,128*scale, kernel_size= 3, stride=1, padding=1)
        self.convolve5 = nn.Conv2d(32*scale, 32*scale, kernel_size=5, stride=1, padding=2 )
        self.max = nn.MaxPool2d(3,stride=1,padding=1)
        
        
        self.bn1 = nn.BatchNorm2d(32*scale)
        self.bn2 = nn.BatchNorm2d(128*scale)
        self.bn3 = nn.BatchNorm2d(64*scale)
        
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self,x):
        y1 = self.bn3(self.relu(self.convolve111(x)))
        y13 = self.bn2(self.relu(self.convolve11(x)))
        y15 = self.bn1(self.relu(self.convolve1(x)))
        ymax = self.max(x)
        
        z1 = self.bn2(self.relu(self.convolve3(y13)))
        z2 = self.bn1(self.relu(self.convolve5(y15)))
        zmax = self.bn1(self.relu(self.convolve1(ymax)))      
        
        
        inception = torch.cat([y1,z1], dim=1)
        inception = torch.cat([inception,z2], dim=1)   
        inception = torch.cat([inception,zmax], dim=1) 
        return inception
   

class InceptionNet(nn.Module):
    
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        
        self.conv1 = nn.Conv2d(3,64,kernel_size=3,stride = 1, padding=1)
        self.maxp = nn.MaxPool2d(2,stride=2)
        self.conv2 = nn.Conv2d(64,192,kernel_size=3,stride =1, padding=1)
        self.maxp = nn.MaxPool2d(2,stride=2)
        self.inception1 = InceptionBlockNew(192,1)
        
        self.inception2 = InceptionBlockNew(256,2)
        self.inception3 = InceptionBlockNew(512,2)
        self.inception4 = InceptionBlockNew(512,2)
        
        self.avg = nn.AvgPool2d(kernel_size=5,stride=3,padding=2)
        self.conv3 = nn.Conv2d(512,256,kernel_size=1,stride=1,padding=0)
        self.conv4 = nn.Conv2d(256,128,kernel_size=1,stride=1,padding=0)
        self.conv5 = nn.Conv2d(128,64,kernel_size=1,stride=1,padding=0)
        
        self.batch = nn.BatchNorm2d(64)
        self.batch2 = nn.BatchNorm2d(192)
        self.batch3 = nn.BatchNorm2d(256)
        self.batch4 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,x):
        
        x1 = self.batch(self.relu(self.conv1(x)))
        x2 = self.maxp(x1)
        x3 = self.batch2(self.relu(self.conv2(x2)))
        x4 = self.maxp(x3)
        
        x5 = self.inception1(x4)
        x6 = self.inception2(x5)
        x7 = self.inception3(x6)
        x8 = self.inception4(x7)
        x9 = self.avg(x8)
        x10 = self.batch3(self.relu(self.conv3(x9)))
        x11 = self.batch4(self.relu(self.conv4(x10)))
        
        
        return x11

class ChestXrayClassifier( nn.Module ):

    def __init__( self, n_class=15, mode='feature-extract',
            backbone='unet', pooling_type='max' ):
        super().__init__()
        self.backbone = None
        self.pool = None
        if backbone  == 'resnet50':
            self.backbone = ResNet50( mode )
        
        elif backbone == 'unet':
            self.backbone = UNet(mode)
            
        elif backbone == 'brandnet':
            self.backbone = BrandNet(mode)
        
        elif backbone == 'inceptionnet':
            self.backbone = InceptionNet(mode)
            
        else:
            assert False and f'Unrecognized backbone: {backbone}'
        if pooling_type == 'max':
            # TODO: is there a way to figure out 8 automatically?
            self.pool = nn.MaxPool2d( kernel_size=2 )
        else:
            assert False and f'Unrecognized pooling_type: {pooling_type}'
        
        self.classifier = nn.Linear(2097152,15)
        

    def forward( self, x ):
        batch_size, _, H, W = x.shape
        batch_size = len( x )

        # expand images to 3 channels since
        # resnet expects 3 channels
        x = x.expand( batch_size, 3, H, W )

        # resnet50 outputs a volume 8x8x2048
        # when input img_size is 256
        x = self.backbone( x )
        x = x.view(batch_size,1024, 1024, 32)

        # TODO: do we need a transition layer?
        x = self.pool(x)
        x = self.pool(x)
        
        
        

        x = x.view( batch_size, -1 )
        x = self.classifier( x )
        return x
