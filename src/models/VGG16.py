import imp

from LitModel import *
from pytorch_vgg16 import KitModel
from dataset import *



#### VGG16 ####
class LitVGG16Model(LitModel):
    _vgg16_weight_file: str = '/work3/s184399/CMPlaces/places205VGG16-torch/pytorch_vgg16.npy'
    
    def __init__(self):
        super().__init__()   
        self.net = KitModel(weight_file=self._vgg16_weight_file).to(gpu)
        
    def preprocess(self, x):
        return x.cuda()
    
    def forward_no_softmax(self, x):
        xp = self.preprocess(x)
        return self.net.forward_no_softmax(xp)
    
    def forward_no_top(self, x):
        xp = self.preprocess(x)
        return self.net.forward_no_top(xp)
        
    def _configure_optim_train(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    @property
    def top(self):
        return self.net.fc8_1
    