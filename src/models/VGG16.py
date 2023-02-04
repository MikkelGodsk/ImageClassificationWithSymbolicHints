import imp

from hydra import compose, initialize

from src.features.dataset import *
from src.models.LitModel import *
from src.models.pytorch_vgg16 import KitModel


initialize(config_path=os.path.join('..', 'conf'), job_name="data_conf")
cfg = compose(config_name="data_conf")


#### VGG16 ####
class LitVGG16Model(LitModel):
    _vgg16_weight_file: str = os.path.join(
        cfg.data.data_folder,
        cfg.data.cmplaces.data_folder,
        "places205VGG16-torch",
        "pytorch_vgg16.npy",
    )
    # "/work3/s184399/CMPlaces/places205VGG16-torch/pytorch_vgg16.npy"

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
