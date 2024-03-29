import imp

from src.features.dataset import *
from src.models.LitModel import *
from src.models.pytorch_vgg16 import KitModel


#### VGG16 ####
class LitVGG16Model(LitModel):
    # "/work3/s184399/CMPlaces/places205VGG16-torch/pytorch_vgg16.npy"

    def __init__(self, ds_dir):
        super().__init__()
        self._vgg16_weight_file: str = os.path.join(
            ds_dir,
            "places205VGG16-torch",
            "pytorch_vgg16.npy",
        )
        self.net = KitModel(weight_file=self._vgg16_weight_file).to(gpu)
        self.name = "VGG16"
        self.M = 4096

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
