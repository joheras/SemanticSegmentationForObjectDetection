from PIL import Image
import numpy as np
import cv2
from prepare_dataset.generateMask import generateMask
from imutils import paths
import os
from prepare_dataset.extractPatches import extractPatches
from tqdm import tqdm
from config import config
from sklearn.model_selection import train_test_split


# Program to generate the dataset of images from the annotated images.
# For each image, we will create a set of patches of size 128x128 and
# their masks. We suppose that the images are annotated using the PASCAL-VOC
# format.


IMAGES_PATH = config.IMAGES_XML_PATH

DATASET_IMAGES_PATH = config.IMAGES_PATH+"/"
DATASET_LABELS_PATH = config.LABELS_PATH+"/"
PATCHSIZE = config.PATCHSIZE
STEP = config.STEP

imagesPaths = list(paths.list_files(IMAGES_PATH,validExts=(".tif",".jpg")))


for imagePath in imagesPaths:
    imageName = name = imagePath.split(os.path.sep)[-1]
    image = cv2.imread(imagePath)
    label = generateMask(imagePath)

    patches_image = extractPatches(image,PATCHSIZE,STEP)
    patches_label = extractPatches(label, PATCHSIZE, STEP)

    for (i,(im,lab)) in tqdm(enumerate(zip(patches_image,patches_label))):
        cv2.imwrite(DATASET_IMAGES_PATH+str(i)+"_"+imageName,im)
        cv2.imwrite(DATASET_LABELS_PATH + str(i) + "_" + imageName, lab)


def load_image(img, img_sz=None):
    if img_sz:
        return np.array(Image.open(img).resize(img_sz, Image.NEAREST))
    else:
        return np.array(Image.open(img))

def load_label(img):
    return np.array(Image.open(img).convert("L"))

# perform stratified sampling from the training set to build the
# testing split from the training data
img_files = list(paths.list_files(config.IMAGES_PATH,validExts=(".tif",".jpg")))
trg_files = list(paths.list_files(config.LABELS_PATH,validExts=(".tif",".jpg")))

split = train_test_split(img_files, trg_files, random_state=42,
                         test_size=0.25)
(trainPaths, testPaths, trainLabels, testLabels) = split

split = train_test_split(trainPaths, trainLabels, random_state=42,
                         test_size=0.1)
(trainPaths, valPaths, trainLabels, valLabels) = split




imgs = np.stack([load_image(img_file, None) for img_file in trainPaths])
labels = np.stack([load_label(trg_file) for trg_file in trainLabels])
imgs = imgs/255.
labels = labels/255
mu = np.mean(imgs)
print(mu)
# mu = 0.6019167833665388
std = np.std(imgs)
# std = 0.2263702138716035
print(std)
imgs -= mu
imgs /=std
np.save(config.NPY_PATH + '/imgs_train.npy', imgs)
np.save(config.NPY_PATH + '/labels_train.npy', labels)

imgs_val = np.stack([load_image(img_file, None) for img_file in valPaths])
labels_val = np.stack([load_label(trg_file) for trg_file in valLabels])
imgs_val = imgs_val/255.
labels_val = labels_val/255
imgs_val -= mu
imgs_val /=std
np.save(config.NPY_PATH + '/imgs_val.npy', imgs_val)
np.save(config.NPY_PATH + '/labels_val.npy', labels_val)


imgs_test = np.stack([load_image(img_file, None) for img_file in testPaths])
labels_test = np.stack([load_label(trg_file) for trg_file in testLabels])
imgs_test = imgs_test/255.
labels_test = labels_test/255
imgs_test -= mu
imgs_test /=std
np.save(config.NPY_PATH + '/imgs_test.npy', imgs_test)
np.save(config.NPY_PATH + '/labels_test.npy', labels_test)