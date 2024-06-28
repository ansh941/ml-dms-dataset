import torch.nn as nn
from models.resnets import Bottleneck, ResnetDilated
from models.ppm import PPM

class Encoder(ResnetDilated):
    def __init__(self, block, layers, dilate_scale=8):
        super(Encoder, self).__init__(block, layers, dilate_scale=dilate_scale)

class Decoder(PPM):
    def __init__(self, num_class, fc_dim, use_softmax):
        super(Decoder, self).__init__(num_class, fc_dim, use_softmax)

class DMS(nn.Module):
    def __init__(self, resnet_block=Bottleneck, resnet_layers=[3,4,6,3], resnet_dilate_scale=8, num_class=46, fc_dim=2048, use_softmax=True):
        super(DMS, self).__init__()
        self.encoder = Encoder(block=resnet_block, layers=resnet_layers, dilate_scale=resnet_dilate_scale)
        self.decoder = Decoder(num_class=num_class, fc_dim=fc_dim, use_softmax=use_softmax)

    def return_feature(self, x):
        encoded = self.encoder(x)
        return encoded
        
    def forward(self, x, seg_size=None):
        encoded = self.encoder(x)
        decoded = self.decoder([encoded], seg_size)
        return decoded
