import numpy as np


def extractPatches(image,patchSize,step):
    (patchW, patchH) = patchSize
    (stepW,stepH) = step
    (h,w) = image.shape[:2]
    patches = []
    startY = 0
    while(startY<h):
        startX = 0
        while(startX<w):
            if (startY+patchH>h) or (startX+patchW>w):

                im = image[startY:startY + patchH, startX:startX + patchW]
                if len(im.shape)==2:
                    nim = np.zeros((patchH, patchW))
                else:
                    nim = np.zeros((patchH,patchW,im.shape[2]))
                nim[0:im.shape[0],0:im.shape[1]]=im
                patches.append(nim)
            else:
                patches.append(image[startY:startY+patchH,startX:startX+patchW])
            startX += stepW
        startY += stepH
    return patches
















