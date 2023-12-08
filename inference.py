import tensorflow as tf
import cv2 as cv
import numpy as np
from yolo import yolov1, yolov1_head, yolov1_loss

classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
fname = 'D:\\programming\\MLDL\\dataset\\pascal2012\\VOCdevkit\\VOC2012\\JPEGImages\\2008_004321.jpg'
sample = cv.imread(fname)
origsize = np.shape(sample)
hw_orig = np.shape(sample)[:2]
sample = cv.resize(sample, (224,224))

sample = np.expand_dims(sample, axis=0).astype(np.float32)
inputs_ts = tf.keras.Input([224,224,3])
outputs_ts = yolov1(S=7,B=2,C=20)(inputs_ts)
yolo_model = tf.keras.Model(inputs=inputs_ts, outputs=outputs_ts)
yolo_model.load_weights('model/full_train1.hdf5')
result = yolo_model.predict(sample)
yolohead = yolov1_head(hw_orig)
boxes = yolohead(result)
sample = np.squeeze(sample, axis=0).astype(np.uint8)
sample = cv.resize(sample, hw_orig[::-1])
for box in boxes:
    
    c = box[0]
    xmin = int(np.clip(box[1], 0,1)*hw_orig[1]) 
    ymin = int(np.clip(box[2], 0,1)*hw_orig[0]) 
    xmax = int(np.clip(box[3], 0,1)*hw_orig[1]) 
    ymax = int(np.clip(box[4], 0,1)*hw_orig[0]) 
    print(box.shape, xmin, ymin, xmax, ymax)
    cv.rectangle(sample, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
    cv.rectangle(sample, (xmin, ymin+20), (xmin+70, ymin), (0, 255, 0), -1)
    
    cv.putText(sample, str(classes_name[int(c.numpy())]), (xmin, ymin+10),
               cv.FONT_HERSHEY_PLAIN,
               1, (0, 0, 0), 1, cv.LINE_AA)
    cv.putText(sample, str(float(box[5])), (xmin, ymin-20),
               cv.FONT_HERSHEY_PLAIN,
               1, (0, 255, 0), 1, cv.LINE_AA)
    

cv.imshow('666', sample.astype(np.uint8))
cv.waitKey(0)