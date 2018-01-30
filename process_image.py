import cv2
from prepare_dataset.extractPatches import extractPatches
import numpy as np
import imutils
from config import config
from segmentation_models.tiramisuModel import TiramisuModel
from segmentation_models.unetModel import UnetModel

IMAGEPATH = "/home/jonathan/Escritorio/Estomas/4042.2.A4.tif"
PREDICTIONPATH = "prediction_20000_4042.2.A4.jpg"
WEIGHTS = "output/"
PATCHSIZE = config.PATCHSIZE
STEP = config.STEP

# Loading the model
if config.MODEL=="Tiramisu":
    segmentationModel = TiramisuModel(config.MU,config.STD,(config.IMAGE_HEIGHT,config.IMAGE_WIDTH,config.CHANNELS))
if config.MODEL=="Unet":
    segmentationModel = UnetModel(config.MU,config.STD,(config.IMAGE_HEIGHT,config.IMAGE_WIDTH,config.CHANNELS))
else:
    raise(ValueError("Not valid model"))

segmentationModel.setWeights(WEIGHTS)


image = cv2.imread(IMAGEPATH)
patches = extractPatches(image,PATCHSIZE,STEP)
print("[Prediction...]")
preds = segmentationModel.predict_images(patches)
(patchW, patchH) = PATCHSIZE
(stepW, stepH) = STEP
(h, w) = image.shape[:2]
mask = np.zeros((image.shape[0],image.shape[1]))
startY = 0
i=0
print("[Generating the mask...]")
while (startY < h):
    startX = 0
    while (startX < w):
        if (startY + patchH > h) or (startX + patchW > w):
            mask[startY:min(startY + patchH, h), startX:min(startX + patchW, w)] = \
                preds[i][0:128-max(0,(startY + patchH -h)),0:128-max(0,(startX + patchW - w))]
        else:
            mask[startY:startY + patchH, startX:startX + patchW] = preds[i]
        print("{}/{}".format(i, len(preds)))
        i+=1
        startX += stepW
    startY += stepH



print("[Saving the mask and showing the result...]")
cv2.imwrite("mask.jpg",mask)
mask = cv2.imread("mask.jpg")
mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

for c in cnts:
    (x,y,w,h) = cv2.boundingRect(c)
    if h > 10 and w > 10:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow("im",image)
cv2.waitKey(0)
cv2.imwrite(PREDICTIONPATH,image)