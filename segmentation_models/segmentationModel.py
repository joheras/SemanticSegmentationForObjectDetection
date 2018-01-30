import numpy as np

class SegmentationModel:
    def __init__(self):
        self.model = None
        self.weights = None
        self.mu = None
        self.std = None

    def getModel(self):
        return self.model

    def setWeights(self,weights):
        self.weights= weights

    def predict_images(self,images):
        if self.weights is None:
            raise ValueError("The weights must be initialized with setWeights")
        else:
            imgs = np.stack(images)
            imgs = imgs / 255.
            imgs -= self.mu
            imgs /= self.std

            preds = self.model.predict(imgs)
            preds = [pred.reshape(128, 128, 2)[:, :, 1] for pred in preds]
            return preds

    def predict_image(self,image):
        return self.predict_images([image])[0]
        