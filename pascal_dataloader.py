import numpy as np
import glob
import random
import math
import tensorflow as tf
from PIL import Image
import cv2.cv2 as cv

class pascal_dataloader:
    def __init__ (self, dataset_path, batchsize, size = [224,224], mode=0):
        self.dataset_path = dataset_path
        self.image_path = dataset_path + '\\VOCdevkit\\VOC2012\\JPEGImages\\'
        self.txt_sample_path = dataset_path + '\\Data\\'
        self.mode = mode
        if mode == 0:
            d_set_name = '\\train.txt'
        elif mode == 1:
            d_set_name = '\\test.txt'
        else:
            raise ValueError(f'mode must be 0 or 1')
        #read dataset from disk
        self.data_set = open(self.dataset_path + d_set_name ,'r').read().split('\n')[:-1]

        #shuffle dataset
        random.shuffle(self.data_set)

        #yolov1 model parameter
        self.S = 7
        self.B = 1
        self.size = size
        self.classes = 20
        self.cell_len = np.array([x/self.S for x in self.size])
        self.batchsize = batchsize
    def __getfname(self):
        return [random.randint(0,len(self.data_set)-1) for _ in range(self.batchsize)]

    def read(self):
        '''
            從字串中分割所有物件的資訊
        '''
        def split_object(line):
            return [float(x) for x in line.split(' ')]
        samplename = self.__getfname()
        
        samples = []
        labels = []
        for s in samplename:
            fname = self.data_set[s]
            sample = cv.imread(fname).astype(np.float32)
            sample = cv.resize(sample,(224,224))

            label = np.zeros([self.S,self.S, 5*self.B+self.classes])
            with open(self.txt_sample_path + fname.split('\\')[-1].split('.')[0] + '.txt', mode='r') as f:
                #get label from string , (class, x, y, w, h)    
                objects = [split_object(x) for x in f.readlines()]
            objects = np.stack(objects, axis=0)
            for o in objects:
                classes = o[0].astype(np.int8)
                xy = np.array(o[1:3])

                wh = np.array(o[3:5])
                ss = np.array([self.S,self.S])
                grid_index = np.floor(xy * ss).astype(np.int8) 
                
                label[grid_index[1], grid_index[0], classes] = 1 #class 
                label[grid_index[1], grid_index[0], 20:22] = xy*ss -grid_index #xy
                label[grid_index[1], grid_index[0], 22:24] = np.log(wh*7) #wh
                label[grid_index[1], grid_index[0], -1] = 1 #conf

            samples.append(sample)
            labels.append(label)
        samples = tf.stack(samples, axis = 0)
        labels = tf.stack(labels, axis = 0)
        
        return samples.numpy(), labels.numpy()

    def generator(self):
        while True:
            yield self.read()

    def __call__(self):
        return self.read()


def test_iou(sample, label):
    feature_hw = [7,7]
    grid_h, grid_w = tf.meshgrid( range(feature_hw[0]), range(feature_hw[1]))
    grid_h = tf.transpose(grid_h)
    grid_h = tf.reshape(grid_h, [1,7,7,1])
    grid_w = tf.reshape(grid_w, [1,7,7,1])
    grid_w = tf.transpose(grid_w)
    grid_wh= tf.cast(tf.reshape(tf.concat([grid_w,grid_h], axis=-1),[1,7,7,2]) , tf.float32) 

    orig_img_size = np.array([500,375])

    label_mask = label[...,-1] == 1 
    label_masked =tf.cast(tf.boolean_mask(label, label_mask), tf.float32) 
    grid_masked = tf.boolean_mask(grid_wh, label_mask)
    print(grid_masked)
    
    xywh = label_masked[...,24:28]
    classes = tf.cast(tf.argmax(label_masked[...,:20], axis=-1), tf.float32) 
    conf = label_masked[...,-1]

    xy= (xywh[...,:2] + grid_masked)/7 * orig_img_size
    wh= xywh[...,2:] * orig_img_size
    xy1 = xy - wh/2
    xy2 = xy + wh/2
    new_box = tf.concat([tf.reshape(classes, [-1,1]), xy1, xy2, tf.reshape(conf, [-1,1])], axis=-1)

    print(new_box)
    xywh = tf.concat([xy,wh], axis=-1)

    gtx1 = 45 
    gtx2 = 244
    gty1 = 122
    gty2 = 303
    
    gtw = gtx2 - gtx1
    gth = gty2 - gty1
    gtx = gtx1 + gtw/2
    gty = gty1 + gth/2

    gtbox = tf.reshape(tf.constant([gtx,gty,gtw,gth]), [1,4]) 
    print(gtbox)
    print(xywh)
    test_iou = calc_iou(xywh, gtbox)
    print(test_iou)
def calc_iou(boxes1, boxes2):
        b1_xy = boxes1[...,:2]
        b1_wh = boxes1[...,2:]
        b1_wh_half = boxes2[...,2:] /2 
        b1_min = b1_xy - b1_wh_half
        b1_max = b1_xy + b1_wh_half

        b2_xy = boxes2[...,:2]
        b2_wh = boxes2[...,2:]
        b2_wh_half = boxes2[...,2:] /2
        b2_min = b2_xy - b2_wh_half
        b2_max = b2_xy + b2_wh_half

        inter_min = tf.maximum(b1_min,b2_min)
        inter_max = tf.minimum(b1_max,b2_max)
        inter_wh = tf.maximum(inter_max-inter_min, tf.zeros_like(inter_max))

        inter_area = inter_wh[...,0] * inter_wh[...,1]
        b1_area = b1_wh[...,0] * b1_wh[...,1]
        b2_area = b2_wh[...,0] * b2_wh[...,1]
        union_area = b1_area + b2_area - inter_area

        iou = tf.clip_by_value(tf.math.divide_no_nan(inter_area,union_area), 0, 1) 

        return iou
def grid():
    h = np.arange(0,7)
    h = np.expand_dims(h, axis=0)
    h = np.tile(h, [7,1])
    h = np.transpose(h)
    w = np.arange(0,7)
    w = np.expand_dims(w, axis=0)
    w = np.tile(w, [7,1])
    grid = np.stack([w,h], axis=-1)
    return grid

def main():
    '''
        test
    '''
    dataset_path = 'E:\\dataset\\pascal2012'
    
    dataloader = pascal_dataloader(dataset_path, 1, mode=0)
    sample,label = dataloader.read()
    sample = sample[0,:,:,:]
    orig_size = np.shape(sample)[:2]
    for i in range(7):
        for j in range(7):
            if label[0,i,j,-1] != 1:
                continue
            c = np.max(label[0,i,j,:20])
            c_list = list(label[0,i,j,:20])
            c = c_list.index(c)

            xywh = label[0,i,j,20:24]
            print(c,xywh)
            xy = (xywh[:2] + np.array([j,i]))/7
            wh = np.exp(xywh[2:])/7
            x = xy[0]
            y = xy[1]
            w = wh[0]
            h = wh[1]

            xmin = int((x-w/2)*orig_size[1])
            ymin = int((y-h/2)*orig_size[0])
            xmax = int((x+w/2)*orig_size[1])
            ymax = int((y+h/2)*orig_size[0])

            print(xy,wh)
            
            cv.rectangle(sample, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv.rectangle(sample, (xmin, ymin+30), (xmin+60, ymin), (255, 255, 255), -1)
            cv.putText(sample, str(c), (xmin, ymin+20),
                            cv.FONT_HERSHEY_PLAIN,
                            1, (0, 0, 255), 1, cv.LINE_AA)
            #cv.circle(sample,(x,y), 3, (255, 0, 0), -1)
    
    cv.imshow('666', sample.astype(np.uint8))
    cv.waitKey(0)

    #test_iou(sample, label)
if __name__ == "__main__":
    main()