class Unet(nn.Module):
    def __init__(self,unet,add_dropouts = False):
        super().__init__()
        #unpack layers of pretrained model
        self.encoder1 = unet.encoder1
        self.pool1 = unet.pool1
        self.encoder2 = unet.encoder2
        self.pool2 = unet.pool2
        self.encoder3 = unet.encoder3
        self.pool3 = unet.pool3
        self.encoder4 = unet.encoder4
        self.pool4 = unet.pool4

        self.bottleneck = unet.bottleneck

        self.upconv4 = unet.upconv4
        self.decoder4 = unet.decoder4
        self.upconv3 = unet.upconv3
        self.decoder3 = unet.decoder3
        self.upconv2 = unet.upconv2
        self.decoder2 = unet.decoder2
        self.upconv1 = unet.upconv1
        self.decoder1 = unet.decoder1

        self.conv = unet.conv



    def forward(self, x):
        #define path

        #encoding
        e1 = self.encoder1(x)

        e2 = self.encoder2(self.pool1(e1))

        e3 = self.encoder3(self.pool2(e2))

        e4 = self.encoder4(self.pool3(e3))

        #bottleneck
        bneck = self.bottleneck(self.pool4(e4))

        #decoding
        d4 = self.upconv4(bneck)
        d4 = torch.cat((d4,e4),dim = 1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3,e3),dim = 1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2,e2),dim = 1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1,e1),dim = 1)
        d1 = self.decoder1(d1)

        res = torch.sigmoid(self.conv(d1))

        return res

class Unet_with_aux_loss_tanh(nn.Module):
    def __init__(self,unet,add_dropouts = False):
        super().__init__()
        #unpack layers of pretrained model

        #encoding
        self.encoder1 =  unet.encoder1
        self.pool1 = unet.pool1
        self.encoder2 =  unet.encoder2
        self.pool2 = unet.pool2
        self.encoder3 =  unet.encoder3
        self.pool3 = unet.pool3
        self.encoder4 = unet.encoder4
        self.pool4 = unet.pool4

        #bottleneck
        self.bottleneck = unet.bottleneck

        #decoding
        self.upconv4 = unet.upconv4
        self.decoder4 =  unet.decoder4
        self.upconv3 = unet.upconv3
        self.decoder3 =  unet.decoder3
        self.upconv2 = unet.upconv2
        self.decoder2 = unet.decoder2
        self.upconv1 = unet.upconv1
        self.decoder1 =  unet.decoder1

        self.conv = unet.conv
        self.poolLayer = nn.AvgPool2d(kernel_size = (16,16), stride= (16,16), ceil_mode = False)
        self.tanh = nn.Tanh()



    def forward(self, x):
        #forward pass
        e1 = self.encoder1(x)

        e2 = self.encoder2(self.pool1(e1))

        e3 = self.encoder3(self.pool2(e2))

        e4 = self.encoder4(self.pool3(e3))

        bneck = self.bottleneck(self.pool4(e4))

        d4 = self.upconv4(bneck)
        d4 = torch.cat((d4,e4),dim = 1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3,e3),dim = 1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2,e2),dim = 1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1,e1),dim = 1)
        d1 = self.decoder1(d1)

        res = torch.sigmoid(self.conv(d1))

        pooled_res = self.poolLayer(res)
        pooled_res = (self.tanh(pooled_res-0.25)+1)/2

        return res,pooled_res
