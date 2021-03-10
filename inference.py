import tensorflow as tf
import cv2.cv2 as cv
import numpy as np

fname = 'E:\\dataset\\pascal2012\\VOCdevkit\\VOC2012\\JPEGImages\\2011_005061.jpg'
sample = cv.imread(fname)

origsize = np.shape(sample)
hw_orig = np.shape(sample)[:2]
sample = cv.resize(sample, (224,224))

sample = np.expand_dims(sample, axis=0).astype(np.float32)

inputs_ts = tf.keras.Input([224,224,3])
outputs_ts = yolov1(S=7,B=1,C=20)(inputs_ts)
yolo_model = tf.keras.Model(inputs=inputs_ts, outputs=outputs_ts)
yolo_model.load_weights('model/full_train.hdf5')
result = yolo_model.predict(sample)

yolohead = yolov1_head(hw_orig)
boxes = yolohead(result)
sample = np.squeeze(sample, axis=0).astype(np.uint8)
sample = cv.resize(sample, hw_orig[::-1])
print(boxes)
for box in boxes:
    c = box[0]
    xmin = int(np.clip(box[1], 0,1)*hw_orig[1]) 
    ymin = int(np.clip(box[2], 0,1)*hw_orig[0]) 
    xmax = int(np.clip(box[3], 0,1)*hw_orig[1]) 
    ymax = int(np.clip(box[4], 0,1)*hw_orig[0]) 
    
    cv.rectangle(sample, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    #cv.rectangle(sample, (xmin, ymin-30), (xmin+60, ymin), (255, 255, 255), -1)
    #cv.putText(sample, str(c.numpy()), (xmin, ymin-20),
    #            cv.FONT_HERSHEY_PLAIN,
    #            1, (0, 0, 255), 1, cv.LINE_AA)

cv.imshow('666', sample.astype(np.uint8))
cv.waitKey(0)