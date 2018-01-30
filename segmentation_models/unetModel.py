from segmentationModel import SegmentationModel
from models.unet import unet_jon

class UnetModel(SegmentationModel):

    def __init__(self,mu,std,inputShape=(128, 128, 3)):
        super(UnetModel,self).__init__()
        self.model = unet_jon(inputShape)
        self.mu = mu
        self.std = std


