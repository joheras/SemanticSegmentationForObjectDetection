import threading
import numpy as np
from numpy import random
from keras.utils import to_categorical
import cv2
from imutils import rotate

class BatchIndices(object):
    """
    Generates batches of shuffled indices.
    # Arguments
        n: number of indices
        bs: batch size
        shuffle: whether to shuffle indices, default False

    """

    def __init__(self, n, bs, shuffle=False):
        self.n, self.bs, self.shuffle = n, bs, shuffle
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.idxs = (np.random.permutation(self.n)
                     if self.shuffle else np.arange(0, self.n))
        self.curr = 0

    def __next__(self):
        with self.lock:
            if self.curr >= self.n: self.reset()
            ni = min(self.bs, self.n - self.curr)
            res = self.idxs[self.curr:self.curr + ni]
            self.curr += ni
            return res

class segm_generator(object):
    """
    Generates batches of sub-images.
    # Arguments
        x: array of inputs
        y: array of targets
        bs: batch size
        out_sz: dimension of sub-image
        train: If true, will shuffle/randomize sub-images
        waldo: If true, allow sub-images to contain targets.
    """
    def __init__(self, x, y, bs=64, out_sz=(128,128), train=True, waldo=True):
        self.x, self.y, self.bs, self.train = x,y,bs,train
        self.waldo = waldo
        self.n = x.shape[0]
        self.ri, self.ci = [], []
        for i in range(self.n):
            ri, ci, _ = x[i].shape
            self.ri.append(ri), self.ci.append(ci)
        self.idx_gen = BatchIndices(self.n, bs, train)
        self.ro, self.co = out_sz
        self.ych = self.y.shape[-1] if len(y.shape)==4 else 1

    def get_slice(self, i,o):
        start = random.randint(0, 10) if self.train else (i-o)
        return slice(start, start+o)

    def get_item(self, idx):
        slice_r = self.get_slice(self.ri[idx], self.ro)
        slice_c = self.get_slice(self.ci[idx], self.co)
        x = self.x[idx][slice_r, slice_c]
        y = self.y[idx][slice_r, slice_c]
        if self.train and (random.random()>0.5):
            y = y[:,::-1]
            x = x[:,::-1]
        if not self.waldo and np.sum(y)!=0:
            return None
        return x, to_categorical(y, num_classes=2)

    def __next__(self):
        idxs = self.idx_gen.__next__()
        items = []
        for idx in idxs:
            item = self.get_item(idx)
            if item is not None:
                items.append(item)
        if not items:
            return None
        xs,ys = zip(*tuple(items))
        return np.stack(xs), np.stack(ys)

def seg_gen(x, y, bs=64, out_sz=(128,128), train=True, waldo=True):
    """
    Generator wrapper on iterators for python 2 compatibility.
    """
    sg = segm_generator(x=x, y=y, bs=bs, out_sz = out_sz ,train=train, waldo=waldo)
    while True:
        yield sg.__next__()

def seg_gen_mix(x1, y1, tot_bs=4):

    while True:
        n=random.randint(0, x1.shape[0]-tot_bs)
        items  = np.zeros((tot_bs,128*128,2))
        for i,y in enumerate(y1[n:n+tot_bs]):

            items[i]=to_categorical(y.reshape(128*128), num_classes=2)
        yield x1[n:n+tot_bs],items


def augment_operation(x,y):
    n = random.randint(0, 5)
    if(n==0):
        newx = cv2.flip(x,0)
        newy = cv2.flip(y,0)
    if(n==1):
        newx = cv2.flip(x, 1)
        newy = cv2.flip(y, 1)
    if(n==2):
        newx = cv2.flip(x, -1)
        newy = cv2.flip(y, -1)
    if(n==3):
        m = random.randint(0,360)
        newx = rotate(x,m)
        newy = rotate(y,m)
    if(n==4):
        newx = x
        newy = y
    return (newx,newy)

def seg_gen_mix_augment(x1, y1, tot_bs=4):

    while True:
        n=random.randint(0, x1.shape[0]-tot_bs)
        items1 = np.zeros(x1[n:n+tot_bs].shape)
        items2  = np.zeros((tot_bs,128*128,2))
        for i,(x,y) in enumerate(zip(x1[n:n+tot_bs],y1[n:n+tot_bs])):
            (x,y)=augment_operation(x,y)
            items1[i]=x
            items2[i]=to_categorical(y.reshape(128*128), num_classes=2)
        yield items1,items2

