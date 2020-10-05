import os
import numpy as np
import tensorflow as tf
import glob
import xml.etree.ElementTree as ET
import pascalvoc_AnnoParse
import cv2
import sys
import time
import random
from multiprocessing import Pool

np.set_printoptions(threshold=sys.maxsize)
classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

class pascalDataloader():
    def __init__(self,labelpath, batchsize):
        self.labelPath = glob.glob(labelpath+'\\*.txt')
        self.labelPath_len = len(self.labelPath)
        self.img_basePath = 'E:/pascal2012/VOCdevkit/VOC2012/JPEGImages'
        self.S = 7
        self.B = 2
        self.imageHeight = 448
        self.imageWidth = 448
        self.classes = 20
        self.batchsize = batchsize

    def rotatePoint(self, xy, img_cx, img_cy, angle):
        '''
        對單個點做rotation transform
        x' = cos(θ) * x - sin(θ) * y
        y' = sin(θ) * x + cos(θ) * y
        '''    
        x ,y = xy
        cosa= np.cos(angle*np.pi/180)
        sina= np.sin(angle*np.pi/180)
        newx= img_cx + cosa * (x -img_cx) - sina * (y -img_cy)
        newy = img_cy + sina * (x - img_cx)  + cosa * (y -img_cy)
        return newx,newy

    def RandomRotate(self, image, bboxes):
        angle = random.randint(-45,45)
        image = tfa.image.rotate(image,-angle*np.pi/180)
        img_w,img_h = tf.shape(image)[1],tf.shape(image)[0]
        img_cx,img_cy = img_w/2, img_h/2
        #找新中心點
        for idx,i in enumerate(bboxes):
            oh=i[1]
            ow=i[2]
            #原本的中心
            ox,oy = i[3],i[4]
            #找中心
            newx,newy= rotatePoint((ox,oy),img_cx,img_cy, angle)
            #找角落
            corner1 = (ox - ow//2,oy + oh//2)
            corner2 = (ox + ow//2,oy + oh//2)
            corner3 = (ox - ow//2,oy - oh//2)
            corner4 = (ox + ow//2,oy - oh//2)
            nc1_x, nc1_y = rotatePoint(corner1,img_cx,img_cy, angle)
            nc2_x, nc2_y = rotatePoint(corner2,img_cx,img_cy, angle)
            nc3_x, nc3_y = rotatePoint(corner3,img_cx,img_cy, angle)
            nc4_x, nc4_y = rotatePoint(corner4,img_cx,img_cy, angle)

            new_minx = np.clip(min([nc1_x,nc2_x,nc3_x,nc4_x]),0,img_w) 
            new_maxx = np.clip(max([nc1_x,nc2_x,nc3_x,nc4_x]),0,img_w)
            new_miny = np.clip(min([nc1_y,nc2_y,nc3_y,nc4_y]),0,img_h)
            new_maxy = np.clip(max([nc1_y,nc2_y,nc3_y,nc4_y]),0,img_h)
            
            new_h ,new_w= (new_maxy  - new_miny) , (new_maxx - new_minx)

            bboxes[idx, 1:5] = [new_h,new_w,newx,newy]
            
        return image, bboxes

    def crop_wrap(self, image, label):
        cx = random.randint(-30,30)
        cy = random.randint(-30,30)

    def flip_v(self,image, bboxes):
        image = tf.image.flip_up_down(image)
        img_h = tf.shape(image)[0]
        #c h w x y
        bboxes[:,4] = img_h - bboxes[:,4]

        return image, bboxes
            
    def flip_h(self,image, bboxes):
        image = tf.image.flip_left_right(image)
        img_w = tf.shape(image)[1]
        #c h w x y
        bboxes[:,3] = img_w - bboxes[:,3]
        return image, bboxes

    def read(self, labelpath):
        #because each bbox predict one object, hence the ground truth shape is s*s*(b+classes)

        #read image and resize
        image = tf.io.read_file(self.img_basePath + '\\' + labelpath.split('\\')[-1].split('.')[0] + '.jpg')
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        with open(labelpath,'r') as f:
            for l in f:
                #0:c 1:h 2:w 3:x 4:y
                l = [int(x) for x in l.split(' ')] 
                bboxes.append(l)
        bboxes = np.stack(bboxes, axis=0)

        #TODO: DATA AGUMENTATION BBOX也必須跟著改

        #垂直翻轉
        if random.randint(0,1):
            image, bboxes = self.flip_v(image, bboxes)
        #水平翻轉
        if random.randint(0,1):
            image, bboxes = self.flip_h(image, bboxes)
        #隨機旋轉
        if random.randint(0,1):
            image, bboxes = self.RandomRotate(image, bboxes)
        #TODO: crop wrap


        #建立label
        label = np.zeros(shape=[7,7, 25])
        for l in bboxes :
            grid_x = int(float(l[3])/64.0)-1
            grid_y = int(float(l[4])/64.0)-1
            #0~19 :class, 20~23 : h w x y, 24 : confidence
            label[grid_x,grid_y, l[0]] = 1
            label[grid_x,grid_y, 20:24] = l[1:5]
            label[grid_x,grid_y, 24] = 1
            
        return image, label

    def __next__(self):
        
        while True:
            random.shuffle(self.labelPath)
            start =0
            end = self.batchsize
            while start < self.labelPath_len:
                x=[]
                y=[]
                for l in self.labelPath[start:end]:
                    x1,y1 = self.read(l)
                    x.append(x1)
                    y.append(y1)
                x = tf.stack(x,axis=0)
                yield x,y
                start += self.batchsize
                end += self.batchsize
if __name__ == "__main__":
    traingen = pascalDataloader('E:\pascal2012\Data',16)

    x,y= next(traingen.__next__())