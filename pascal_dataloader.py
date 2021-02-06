import numpy as np
import glob
import random
import math
import tensorflow as tf


class pascal_dataloader:
    def __init__ (self, dataset_path, batchsize, size = [448,448], mode=0):
        self.dataset_path = dataset_path
        self.image_path = dataset_path + '\\VOCdevkit\\VOC2012\\JPEGImages\\'
        self.txt_sample_path = dataset_path + '\\Data\\'

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
        self.B = 2
        self.size = size
        self.classes = 20
        self.precell_len = np.array([x/self.S for x in self.size])
        self.batchsize = batchsize
    def __getfname(self):
        return [random.randint(0,len(self.data_set)-1) for _ in range(self.batchsize)]
    def read(self):
        '''
            從字串中分割所有物件的資訊
        '''
        def split_object(line):
            return [int(x) for x in line.split(' ')]
        samplename = self.__getfname()
        samples = []
        labels = []
        for s in samplename:
            fname = self.data_set[s]
            sample = tf.io.read_file(fname)
            sample = tf.image.decode_jpeg(sample, channels=3)
            sample = tf.image.convert_image_dtype(sample, tf.float32)
            sample = tf.image.resize(sample,self.size)


            label = np.zeros([self.S,self.S, 5*(self.B)+self.classes])
            with open(self.txt_sample_path + fname.split('\\')[-1].split('.')[0] + '.txt', mode='r') as f:
                #get label from string , (class, x, y, w, h)    
                objects = [split_object(x) for x in f.readlines()]
            objects = np.stack(objects, axis=0)
            for o in objects:
                xy = np.array(o[1:3])
                wh = np.array(o[3:5])
                grid_y, grid_x = np.ceil(xy/self.precell_len).astype(np.int8)-1
                rxry= np.array([grid_y+1,grid_x+1])*self.precell_len
                
                label[grid_y, grid_x, o[0]] = 1 #class
                
                label[grid_y, grid_x, 20:22] = (xy-rxry)/rxry #xy
                label[grid_y, grid_x, 22:24] = (wh)/self.size #wh
                label[grid_y, grid_x, 24:26] = (xy-rxry)/rxry #xy
                label[grid_y, grid_x, 26:28] = (wh)/self.size #wh

                label[grid_y, grid_x, 28] = 1 #conf
                label[grid_y, grid_x, 29] = 1 #conf
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

def main():
    '''
        test
    '''
    dataset_path = 'E:\\dataset\\pascal2012'
    
    dataloader = pascal_dataloader(dataset_path, 16, mode=0)
    sample,label = dataloader.read()
    
if __name__ == "__main__":
    main()