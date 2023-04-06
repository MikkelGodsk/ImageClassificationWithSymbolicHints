# import torch
# import torchvision
from torchvision.models import ResNet50_Weights, resnet50

from src.models.LitModel import *

# from transformers import AutoFeatureExtractor, ResNetForImageClassification


#### ResNet50 ####
class LitResNet50Model(LitModel):
    def __init__(self):
        super().__init__()
        """
        Src: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
            https://pytorch.org/vision/stable/models.html
        """
        # Note: The new version of torch will only run on v100 and not a100.

        self.net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(gpu)
        self.name = "ResNet50"
        self.M = 2048

    def preprocess(self, x):
        return x.cuda()

    def forward_no_softmax(self, x):
        emb = self.forward_no_top(x)
        return self.net.fc(emb)

    def forward_no_top(self, x):
        x = self.preprocess(x)

        # From https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def _configure_optim_train(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    @property
    def top(self):
        return self.net.fc
