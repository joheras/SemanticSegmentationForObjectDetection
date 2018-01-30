from segmentation_models.tiramisuModel import TiramisuModel
from segmentation_models.unetModel import UnetModel
import os

import keras
from keras.layers import Input
from keras.models import Model

from config import config
from models.metrics import mean_iou
from models.tiramisu import *
from utils.utils import *

# Loading the datasets
imgs = np.load(config.NPY_PATH + '/imgs_train.npy')
labels = np.load(config.NPY_PATH + '/labels_train.npy')
imgs_val = np.load(config.NPY_PATH + '/imgs_val.npy')
labels_val = np.load(config.NPY_PATH + '/labels_val.npy')
imgs_test = np.load(config.NPY_PATH + '/imgs_test.npy')
labels_test = np.load(config.NPY_PATH + '/labels_test.npy')

# Computing the frequency of 0s and 1s
freq0 = float(np.sum(labels==0))
freq1 = float(np.sum(labels==1))
print(imgs.shape[0])
gen_mix = seg_gen_mix(imgs, labels)
X, y = next(gen_mix)
print(y.shape)


class_weights = {0:1/freq0, 1:1.}
sample_weights = np.zeros((X.shape[0], X.shape[1] * X.shape[2], 2))
sample_weights.shape
sample_weights[:,:,0] = class_weights[0]
sample_weights[:,:,1] = class_weights[1]


# Loading the model
if config.MODEL=="Tiramisu":
    segmentationModel = TiramisuModel(config.MU,config.STD,(config.IMAGE_HEIGHT,config.IMAGE_WIDTH,config.CHANNELS))
if config.MODEL=="Unet":
    segmentationModel = UnetModel(config.MU,config.STD,(config.IMAGE_HEIGHT,config.IMAGE_WIDTH,config.CHANNELS))
else:
    raise(ValueError("Not valid model"))

model = segmentationModel.getModel()
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop(1e-3), metrics=["accuracy",mean_iou], sample_weight_mode='temporal')

gen_mix = seg_gen_mix(imgs, labels,tot_bs=6)
gen_mix_val = seg_gen_mix(imgs_val,labels_val)

items  = np.zeros((len(labels_test),128*128,2))
for i,y in enumerate(labels_test):
    items[i] = to_categorical(y.reshape(128 * 128), num_classes=2)


for n in np.arange(0,config.EPOCHS,config.EPOCHS_SAVE):
    model.fit_generator(gen_mix,steps_per_epoch=10, epochs=config.EPOCHS_SAVE, verbose=1, class_weight=sample_weights,
                    validation_data=gen_mix_val,validation_steps=10)
    model.save(config.OUTPUT + config.MODEL +"_" + str((n+1)*config.EPOCHS_SAVE) + "epochs.h5")

    file = open(config.MODEL + "_results.txt","a")
    file.write("Evaluation " + str((n+1)*config.EPOCHS_SAVE))
    file.write(str(model.evaluate(imgs_test,items,batch_size=32)))
    file.close()
