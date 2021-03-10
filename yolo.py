import math
import os
import cv2.cv2 as cv
import numpy as np
import tensorflow as tf
import pascal_dataloader


class conv2d_block(tf.Module):
    def __init__(self, filters, kernel_size, strides, padding, leaky_relu_alpha=0.1, \
                                    pool_stride=2, pool_ksize=2,use_leaky_relu=True, use_maxpool=False, name='conv2d_block'):
        super(conv2d_block, self).__init__()
        self.conv2d =tf.keras.layers.Conv2D(filters, 
                                            kernel_size, 
                                            strides, 
                                            padding
                                            )
        self.leaky_relu_alpha = leaky_relu_alpha
        self.pool_stride = pool_stride
        self.use_leaky_relu = use_leaky_relu
        self.use_maxpool = use_maxpool
        self.pool_ksize = pool_ksize
        self.batch_norm = tf.keras.layers.BatchNormalization()
    def __call__(self, inputs):
        x = self.conv2d(inputs)
        x = self.batch_norm(x)
        if self.use_leaky_relu:
            x = tf.nn.leaky_relu(x, alpha=self.leaky_relu_alpha)
        else:
            x = tf.nn.relu(x)
        if self.use_maxpool:
            x = tf.nn.max_pool2d(x, ksize= self.pool_ksize, strides= self.pool_stride, padding='SAME')
        return x

class darknet19(tf.Module):
    def __init__(self, name='darknet19'):
        super(darknet19, self).__init__()
        self.conv_block1 = conv2d_block(64, 7, 2, padding='SAME', use_maxpool=True)

        self.conv_block2 = conv2d_block(192, 2, 1, padding='SAME', use_maxpool=True)

        self.conv_block3 = conv2d_block(128, 3, 1, padding='SAME')
        self.conv_block4 = conv2d_block(256, 3, 1, padding='SAME')
        self.conv_block5 = conv2d_block(128, 3, 1, padding='SAME')
        self.conv_block6 = conv2d_block(256, 3, 1, padding='SAME', use_maxpool=True)
        
        self.conv_block7 = conv2d_block(256, 1, 1, padding='SAME')
        self.conv_block8 = conv2d_block(512, 3, 1, padding='SAME')
        self.conv_block9 = conv2d_block(256, 1, 1, padding='SAME')
        self.conv_block10 = conv2d_block(512, 3, 1, padding='SAME')
        self.conv_block11 = conv2d_block(256, 1, 1, padding='SAME')
        self.conv_block12 = conv2d_block(512, 3, 1, padding='SAME')
        self.conv_block13 = conv2d_block(256, 1, 1, padding='SAME')
        self.conv_block14 = conv2d_block(512, 3, 1, padding='SAME')

        self.conv_block15 = conv2d_block(512, 1, 1, padding='SAME')
        self.conv_block16 = conv2d_block(1024, 3, 1, padding='SAME', use_maxpool=True)

        self.conv_block17 = conv2d_block(512, 1, 1, padding='SAME')
        self.conv_block18 = conv2d_block(1024, 3, 1, padding='SAME')
        self.conv_block19 = conv2d_block(512, 1, 1, padding='SAME')
        self.conv_block20 = conv2d_block(1024, 3, 1, padding='SAME')
        self.conv_block21 = conv2d_block(1024, 3, 1, padding='SAME')
        self.conv_block22 = conv2d_block(1024, 3, 2, padding='SAME')

        self.conv_block23 = conv2d_block(1024, 3, 1, padding='SAME')
        self.conv_block24 = conv2d_block(1024, 3, 1, padding='SAME')
        self.Flatten1 = tf.keras.layers.Flatten()
    def __call__(self, inputs):
        x = self.conv_block1(inputs)

        x = self.conv_block2(x)

        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        
        x = self.conv_block7(x)
        x = self.conv_block8(x)
        x = self.conv_block9(x)
        x = self.conv_block10(x)
        x = self.conv_block11(x)
        x = self.conv_block12(x)
        x = self.conv_block13(x)
        x = self.conv_block14(x)

        x = self.conv_block15(x)
        x = self.conv_block16(x)

        x = self.conv_block17(x)
        x = self.conv_block18(x)
        x = self.conv_block19(x)
        x = self.conv_block20(x)
        x = self.conv_block21(x)
        x = self.conv_block22(x)

        x = self.conv_block23(x)
        x = self.conv_block24(x)

        x = self.Flatten1(x)
        return x

class yolov1(tf.Module):
    def __init__(self, S, B, C):
        super(yolov1, self).__init__(name='yolov1_model')
        self.S = S
        self.B = B
        self.C = C
        self.darknet = darknet19()
        self.dense1 = tf.keras.layers.Dense(self.S*self.S*(self.B*5+self.C), activation='linear')
        self.output_layer = yolov1_conv(self.S,self.B,self.C)
        
    def __call__(self, inputs):
        x = self.darknet(inputs)
        x = self.dense1(x)
        x = self.output_layer(x)
        return x

class yolov1_conv(tf.Module):
    def __init__(self, S,B,C):
        super(yolov1_conv, self).__init__(name='yolov1_conv')
        self.S = S
        self.B = B
        self.C = C

    def __call__(self, inputs):
        inputs = tf.reshape(inputs, [-1, self.S, self.S,(5*self.B+self.C)])
        xywh = inputs[...,self.C:self.C+4*self.B]
        xy = tf.sigmoid(xywh[...,:2])
        wh = xywh[...,2:]
        
        conf = tf.expand_dims(tf.sigmoid(inputs[...,-1]), axis=-1) 
        classes = tf.nn.softmax(inputs[...,:self.C])

        return tf.concat([classes, xy, wh, conf], axis=-1)

class yolov1_loss(tf.Module):
    def __init__(self, batch_size=16, image_size=[224,224], S=7, 
    B=2, C=20, obj_scale=1, noobj_scale=0.5, 
    class_scale=1, coord_scale=5):
        super(yolov1_loss, self).__init__(name='yolov1_loss')
        self.batch_size = batch_size
        self.image_size = image_size
        self.S =S
        self.B =B
        self.C =C
        self.outputsize = [self.batch_size, self.S,self.S, (self.B*5)+self.C]
        self.obj_scale = obj_scale
        self.noobj_scale = noobj_scale
        self.class_scale = class_scale
        self.coord_scale = coord_scale
        self.__name__='yolov1_loss'
    def coordinate_loss(self, pred_xywh, true_xywh):
        cood_loss= tf.reduce_sum(tf.square(pred_xywh-true_xywh), axis=3) 
        return self.coord_scale * cood_loss
        
    def object_loss(self, pred_conf):
        return self.obj_scale * tf.reduce_sum(tf.square(tf.ones_like(pred_conf)- pred_conf),axis=3)

    def noobject_loss(self, pred_conf):
        return self.noobj_scale * tf.reduce_sum(tf.square(pred_conf),axis=3)

    def class_loss(self, pred_classes, true_classes):
        class_loss = self.class_scale * tf.keras.losses.categorical_crossentropy(true_classes, pred_classes)
        return class_loss

    def calc_iou(self, boxes1, boxes2):

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

    def grid(self, pred_ts):
        '''
            pred_ts : batch, s, s, 5*b+c
        '''
        h = np.arange(0,7)
        h = np.expand_dims(h, axis=0)
        h = np.tile(h, [7,1])
        h = np.transpose(h)
        w = np.arange(0,7)
        w = np.expand_dims(w, axis=0)
        w = np.tile(w, [7,1])
        grid = np.stack([w,h], axis=-1)
        grid_wh = tf.constant(grid, tf.float32)
        return grid_wh

    def __call__(self, y_true, y_pred):

        '''
        ground truth batch * 7 * 7 * 30
        |   classes *20 | box1 *4   | box 2 *4  | conf*2  |

        output tensor batch * 7 * 7 * 30
        |   classes *20 | box1 *4   | box 2 *4  | conf*2  |

        '''
        ypred_bbox_offset = tf.reshape(tf.cast(y_pred[...,20:24], tf.float32) , [-1,7,7,4])
        ytrue_bbox_offset = tf.reshape(tf.cast(y_true[...,20:24], tf.float32) , [-1,7,7,4])
                
        respon_mask = tf.reshape(tf.cast(y_true[...,-1] > 0, tf.float32), [-1,7,7]) 
        '''
            confidence loss     coordinate_loss     class_loss      
                positive
                negative
        '''
        coord_loss = self.coordinate_loss(ypred_bbox_offset,ytrue_bbox_offset) * respon_mask

        conf_pred = tf.cast(tf.reshape(y_pred[...,-1],[-1,7,7,1]), tf.float32)
        object_loss = self.object_loss(conf_pred) * respon_mask

        no_object_loss = self.noobject_loss(conf_pred) * (1- respon_mask)

        pred_class = tf.cast(tf.reshape(y_pred[...,:20],[-1,7,7,20]), tf.float32) 
        true_class = tf.cast(tf.reshape(y_true[...,:20],[-1,7,7,20]), tf.float32)
        class_loss = self.class_loss( pred_class,true_class) * respon_mask
            
        loss = tf.reduce_sum((coord_loss + object_loss + no_object_loss + class_loss), axis=[1,2]) 
        return loss

class yolov1_head(tf.Module):
    def __init__(self, 
                 orig_img_size,
                 class_num=20,
                 iou_threshold = 0.5,
                 scores_threshold = 0.4,
                 yolo_img_size=(224,224),
                 name = 'yolov1_head'):
        super(yolov1_head, self).__init__(name=name)
        self.iou_threshold = iou_threshold
        self.scores_threshold = scores_threshold
        self.yolo_img_size = np.array(yolo_img_size)
        self.orig_img_size = np.array(orig_img_size)
        self.class_num = class_num

    def nms(self, boxes):
        pred_box = boxes[...,1:5]
        pred_classes = boxes[...,0]
        pred_scores = boxes[...,-1]
        nms_idx = tf.image.non_max_suppression(pred_box, 
                                               pred_scores,
                                               100,
                                               iou_threshold=self.iou_threshold,
                                               score_threshold=self.scores_threshold)
        pred_box = tf.gather(pred_box, nms_idx)
        pred_classes = tf.gather(pred_classes, nms_idx)
        pred_scores = tf.gather(pred_scores, nms_idx)
        pred_boxes = tf.concat([tf.reshape(pred_classes, [-1,1]), pred_box, tf.reshape(pred_scores, [-1,1])], axis=-1)
        return pred_boxes

    def grid(self, pred_ts):
        '''
            pred_ts : batch, s, s, 5*b+c
        '''
        h = np.arange(0,7)
        h = np.expand_dims(h, axis=0)
        h = np.tile(h, [7,1])
        h = np.transpose(h)
        w = np.arange(0,7)
        w = np.expand_dims(w, axis=0)
        w = np.tile(w, [7,1])
        grid = np.stack([w,h], axis=-1)
        grid_wh = tf.constant(grid, tf.float32)
        return grid_wh

    def preprocess_box(self, inputs_ts):
        feature_hw = tf.cast(tf.shape(inputs_ts)[1:3], tf.float32) 

        pred_classes = tf.reshape(inputs_ts[...,:20], [-1,7,7,20])

        #把xywh換成xmax xmin ymax ymin
        pred_xywh = tf.reshape(inputs_ts[...,20:24], [-1,7,7,4])
        pred_xy = (pred_xywh[...,:2] + self.grid(inputs_ts)) / feature_hw[0]
        pred_wh = np.exp(pred_xywh[...,2:])/7
        pred_x1y1 = pred_xy - pred_wh/2
        pred_x2y2 = pred_xy + pred_wh/2
        pred_box = tf.reshape(tf.concat([pred_x1y1, pred_x2y2], axis=-1),[-1,7,7,4]) 
        #算出scores
        pred_conf = tf.reshape(inputs_ts[...,-1], [-1,7,7,1]) 
        pred_scores = tf.reduce_max(pred_classes * pred_conf,axis=-1)
        pred_classes = tf.cast(tf.argmax(pred_classes, axis=-1), tf.float32) 
        # pred_classes = tf.boolean_mask(pred_classes, pred_score_mask)

        pred_boxes = tf.concat([
                                tf.reshape(pred_classes, [-1,1]),
                                tf.reshape(pred_box,[-1,4]),
                                tf.reshape(pred_scores, [-1,1])
                            ], axis=-1)
        return pred_boxes
    def __call__(self, inputs_ts):    
        pred_boxes = self.preprocess_box(inputs_ts)
        pred_classes = pred_boxes[...,0]
        pred_xywh = pred_boxes[...,1:5]
        pred_scores = pred_boxes[...,-1]
        result_boxes = tf.zeros([0,6])
        print(pred_boxes)
        for i in range(self.class_num):
            class_mask = (pred_classes == i)
            _pred_xywh = tf.boolean_mask(pred_xywh, class_mask)
            _pred_scores = tf.boolean_mask(pred_scores, class_mask)
            _pred_classes = tf.boolean_mask(pred_classes , class_mask)
            if(tf.shape(_pred_classes)[0]==0):
                continue
            __pred_boxes = tf.concat([
                                        tf.reshape(_pred_classes, [-1,1]), 
                                        _pred_xywh, 
                                        tf.reshape(_pred_scores, [-1,1])
                                    ], axis=-1)
            
            result_boxes = tf.concat([result_boxes, self.nms(__pred_boxes)],axis=0 ) 
            
        return result_boxes


def loss_test():
    dataset_path = 'E:\\dataset\\pascal2012'
    d = pascal_dataloader.pascal_dataloader(dataset_path, 8, mode=0)
    image, label = d.read()
    c = tf.random.normal([8,7,7,30],0.5,0.25)
    b = tf.random.normal([8,7,7,30])
    l = label+1
    yololoss = yolov1_loss(batch_size=8)
    loss = yololoss.call(label,label)
    print(loss)
    
if __name__ =='__main__':
    loss_test()