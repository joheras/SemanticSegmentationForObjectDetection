# Program to generate the mask from a PASCALVOC annotation file.

import cv2
import os
import xml.etree.ElementTree as ET
import numpy as np


def generateMask(imagePath):
    image = cv2.imread(imagePath)
    name = imagePath.split(os.path.sep)[-1]
    labelPath = '/'.join(imagePath.split(os.path.sep)[:-1]) + "/"+name[0:name.rfind(".")] + ".xml"
    tree = ET.parse(labelPath)
    root = tree.getroot()
    objects = root.findall('object')
    if(len(objects)<1):
        raise Exception("The xml should contain at least one object")
    boxes = []
    for object in objects:
        category = object.find('name').text
        bndbox = object.find('bndbox')
        x  = int(bndbox.find('xmin').text)
        y = int(bndbox.find('ymin').text)
        h = int(bndbox.find('ymax').text)-y
        w = int(bndbox.find('xmax').text) - x
        boxes.append((category, (x, y, w, h)))

    mask = np.zeros((image.shape[0],image.shape[1],1), dtype="uint8")
    for (category, (x, y, w, h)) in boxes:
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    return mask

# im = cv2.imread("images/307A5_Multifocus Image(1).tif")
# cv2.imshow("image",im)
# cv2.imshow("mask",generateMask("images/307A5_Multifocus Image(1).tif"))
# cv2.waitKey(0)