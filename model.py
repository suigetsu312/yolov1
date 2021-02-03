import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer, BatchNormalization, LeakyReLU

class yolo(Model):
    def __init__(self):
        super(yolo,self).__init__()
        self.imageSize = (448, 448, 3)
        self.num_classes = 20
        self.S = 7
        self.B = 2
        self.batch_size = 16
        #self.weight_decay = 0.01
        self.leaky_relu_alpha = 0.1
        self.obj_scale = 1
        self.noobj_scale = 0.5
        self.coord_scale = 5
        self.class_scale = 1

        self.conf_threshold = .5
        self.iou_threshold = .5
    
    def conv2d_block(self, feature, filters, kernel_size, strides, padding='same'):
        x = Conv2D(filters, kernel_size, strides, padding)(feature)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=self.leaky_relu_alpha)(x)
        return x
    
    def darknet19(self):
        input_layer = InputLayer(input_shape=self.imageSize)
        x = self.conv2d_block(input_layer, 64, 7, 2, 'same')
        x = MaxPool2D(strides=2)(x)

        x = self.conv2d_block(x, 192, 3, 0, 'same')
        x = MaxPool2D(strides=2)(x)

        x = self.conv2d_block(x, 256, 1, 0, 'same')
        x = self.conv2d_block(x, 512, 3, 0, 'same')
        x = self.conv2d_block(x, 512, 1, 0, 'same')
        x = self.conv2d_block(x, 1024, 3, 0, 'same')
        x = MaxPool2D(strides=2)(x)

        for _ in range(4):
            x = self.conv2d_block(x, 256, 1, 0, 'same')
            x = self.conv2d_block(x, 512, 3, 0, 'same')
        x = self.conv2d_block(x, 512, 1, 0, 'same')
        x = self.conv2d_block(x, 1024, 3, 0, 'same')
        x = MaxPool2D(strides=2)(x)

        for _ in range(2):
            x = self.conv2d_block(x, 512, 1, 0, 'same')
            x = self.conv2d_block(x, 1024, 3, 0, 'same')
        x = self.conv2d_block(x, 1024, 3, 0, 'same')
        x = self.conv2d_block(x, 1024, 3, 2, 'same')

        x = self.conv2d_block(x, 1024, 3, 0, 'same')
        x = self.conv2d_block(x, 1024, 3, 0, 'same')

        x = Flatten()(x)
        x = Dense(4096, activation=LeakyReLU(self.leaky_relu_alpha))(x)
        output = Dense(self.S*self.S*(self.B*5+self.num_classes), activation=LeakyReLU(self.leaky_relu_alpha))(x)        
        
        model = tf.keras.Model(inputs=input_layer, outputs= output)
        return model

    def yolo_loss(self, pre, target):

        #該cell有物件
        object_mask = target[...,24] >0
        #該cell無物件
        no_object_mask = tf.cast(1-tf.cast(object_mask, tf.int32), tf.bool)

        object_mask = tf.expand_dims(object_mask,axis=-1)
        object_mask = tf.tile(object_mask, [1,1,1,30])

        pre_objects = tf.reshape(tf.boolean_mask(pre, object_mask),(-1,30))
        target_objects = tf.reshape(tf.boolean_mask(target, object_mask),(-1,30))
        #該cell無物件
        
        no_object_mask = tf.expand_dims(no_object_mask,axis=-1)
        no_object_mask = tf.tile(no_object_mask, [1,1,1,30])

        pre_noobjects = tf.reshape(tf.boolean_mask(pre, no_object_mask),(-1,30))
        target_noobjects = tf.reshape(tf.boolean_mask(target, no_object_mask),(-1,30))

        #box
        pre_box = pre_objects[..., 20:28]
        pre_box = tf.reshape(pre_box, (-1,4))

        pre_conf = pre_objects[...,28:30]
        pre_conf = tf.reshape(pre_conf, (-1,1))

        pre_box = tf.concat([pre_box, pre_conf], -1)

        target_box = target_objects[..., 20:28]
        target_box = tf.reshape(target_box, (-1,4))

        target_conf = target_objects[...,28:30]
        target_conf = tf.reshape(target_conf, (-1,1))

        target_box = tf.concat([target_box, target_conf], -1)
        #class
        pre_class = pre_objects[..., :20]
        target_class = target_objects[..., :20]

        #no object confidence區塊
        pre_noob_conf = pre_noobjects[...,29:30]
        target_noob_conf = target_noobjects[...,29:30]
        #no object confidence loss
        noob_conf_loss = self.noobj_scale * tf.nn.l2_loss(pre_noob_conf, target_noob_conf)
        #class loss
        class_loss = tf.nn.l2_loss(target_class, pre_class)
        
        #coordinate loss
        #計算所有box的iou
        all_box_iou = self.iou(target_box[...,:4], pre_box[...,:4])
        #分開cell中第一個box和第二個box的iou
        print(all_box_iou.numpy())
        all_box_iou = tf.reshape(all_box_iou,(-1,2))
        
        argmax_mask = tf.argmax(all_box_iou, axis=-1)
        argmax_mask = tf.reshape(argmax_mask, [-1,1])
        argmax_mask = tf.tile(argmax_mask, [1,5])
        argmax_mask = tf.cast(argmax_mask, tf.bool)
        not_mask = tf.math.logical_not(argmax_mask)
        argmax_mask = tf.concat([argmax_mask, not_mask], 0)

        all_box_iou = tf.reshape(all_box_iou,(-1,1))
        all_box_iou = tf.tile(all_box_iou,[1,5])
        all_box_iou = tf.cast(all_box_iou,tf.float32)

        #留下具有代表性的box的iou (較大iou的那個)
        all_box_iou = tf.reshape(tf.boolean_mask(all_box_iou, argmax_mask), [-1,5])
        #除掉iou較低的box
        heigher_iou_prebox =  tf.reshape(tf.boolean_mask(pre_box, argmax_mask), [-1,5])
        heigher_iou_targetbox = tf.reshape(tf.boolean_mask(target_box, argmax_mask), [-1,5])


        conf_loss = self.obj_scale * tf.nn.l2_loss(heigher_iou_prebox[-1,4] - all_box_iou[-1,4])

        coordinate_loss = self.coord_scale * (tf.nn.l2_loss(heigher_iou_prebox[:,:2]-heigher_iou_targetbox[:,:2]) + \
                            tf.nn.l2_loss(tf.sqrt(heigher_iou_prebox[:,2:4])-tf.sqrt(heigher_iou_targetbox[:,2:4])))

        
        return (class_loss + noob_conf_loss + conf_loss + coordinate_loss)

    def iou(self, boxes1, boxes2):
        boxes1 = tf.stack([boxes1[:,0]-0.5*boxes1[:,2],boxes1[:,1]-0.5*boxes1[:,3],boxes1[:,0]+0.5*boxes1[:,2],boxes1[:,1]+0.5*boxes1[:,3]],axis=-1)
        boxes2 = tf.stack([boxes2[:,0]-0.5*boxes2[:,2],boxes2[:,1]-0.5*boxes2[:,3],boxes2[:,0]+0.5*boxes2[:,2],boxes2[:,1]+0.5*boxes2[:,3]],axis=-1)

        lu = tf.maximum(boxes1[:,0:2],boxes2[:,0:2])    
        rd = tf.minimum(boxes1[:,2:],boxes2[:,2:])
        intersection = rd-lu

        inter_square = intersection[:,0] * intersection[:,1]
        mask = tf.cast(intersection[:,0] > 0, tf.float32) * tf.cast(intersection[:,1] > 0, tf.float32)

        inter_square = mask * inter_square
        square1 = (boxes1[:,2] - boxes1[:,0]) * (boxes1[:,3] - boxes1[:,1])
        square2 = (boxes2[:,2] - boxes2[:,0]) * (boxes2[:,3] - boxes2[:,1])

        return inter_square/(square1 + square2 - inter_square + 1e-6)

if __name__ == '__main__':
    a = tf.random.normal([3,7,7,30])
    b = tf.random.normal([3,7,7,30])
    yololoss = yolo()
    loss = yololoss.yolo_loss(a,b)
    print(loss.numpy())