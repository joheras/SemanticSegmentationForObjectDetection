from segmentationModel import SegmentationModel
from models.tiramisu import *


class TiramisuModel(SegmentationModel):

    def __init__(self,mu,std,inputShape=(128, 128, 3)):
        super(TiramisuModel,self).__init__()
        img_input = Input(shape=inputShape)
        x = create_tiramisu(2, img_input, nb_layers_per_block=[4, 5, 7, 10, 12, 15], p=0.2, wd=1e-4)
        self.model = Model(img_input, x)
        self.mu = mu
        self.std = std


