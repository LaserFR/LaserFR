import torch
import torch.nn as nn
import torch.nn.functional as F

class SFaceBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, pool=False):
        super(SFaceBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.pool = pool
        if self.pool:
            self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        if self.pool:
            x = self.maxpool(x)
        return x

class SFace(nn.Module):
    def __init__(self, embedding_dim=512):
        super(SFace, self).__init__()
        self.layer1 = SFaceBlock(3, 64, 3, 1, 1, pool=True)
        self.layer2 = SFaceBlock(64, 128, 3, 1, 1, pool=True)
        self.layer3 = SFaceBlock(128, 256, 3, 1, 1, pool=True)
        self.layer4 = SFaceBlock(256, 512, 3, 1, 1, pool=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, embedding_dim, bias=False)
        self.bn = nn.BatchNorm1d(embedding_dim, eps=0.001, momentum=0.995)

    def forward(self, x, with_feature=False):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        features = self.avgpool(x)
        features_flat = torch.flatten(features, 1)
        embeddings = self.fc(features_flat)
        embeddings_bn = self.bn(embeddings)

        if with_feature:
            return embeddings_bn, features, self.fc, self.bn
        else:
            return embeddings_bn

def Inception_v3():
    return SFace(embedding_dim=512)


