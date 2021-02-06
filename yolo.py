import tensorflow as tf
import pascal_dataloader
class conv2d_block(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, leaky_relu_alpha=0.1, \
                                    pool_stride=2, pool_ksize=2,use_leaky_relu=True, use_maxpool=False):
        super(conv2d_block, self).__init__()
        self.conv2d =tf.keras.layers.Conv2D(filters, kernel_size, strides, padding)
        self.leaky_relu_alpha = leaky_relu_alpha
        self.pool_stride = pool_stride
        self.use_leaky_relu = use_leaky_relu
        self.use_maxpool = use_maxpool
        self.pool_ksize = pool_ksize
        self.batch_norm = tf.keras.layers.BatchNormalization()
    def call(self, inputs):
        x = self.conv2d(inputs)
        x = self.batch_norm(x)
        if self.use_leaky_relu:
            x = tf.nn.leaky_relu(x, alpha=self.leaky_relu_alpha)
        else:
            x = tf.nn.relu(x)
        if self.use_maxpool:
            x = tf.nn.max_pool2d(x, ksize= self.pool_ksize, strides= self.pool_stride, padding='SAME')
        return x
class darknet19(tf.keras.layers.Layer):
    def __init__(self):
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
    def call(self, inputs):
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

class yolov1(tf.keras.Model):
    def __init__(self, S, B, C):
        super(yolov1, self).__init__(name='yolov1_model')
        self.S = S
        self.B = B
        self.C = C
        self.darknet = darknet19()
        self.output_layer = tf.keras.layers.Dense(self.S*self.S*(self.B*5+self.C), activation='linear')
    def call(self, inputs):
        x = self.darknet(inputs)
        x = self.output_layer(x)
        return x

class yolov1_loss(tf.keras.losses.Loss):
    def __init__(self, batch_size=16, image_size=[448,448], S=7, B=2, C=20, obj_scale=1, noobj_scale=0.5, class_scale=1, coord_scale=5):
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
    def coordinate_loss(self, pred_box, true_box):
        pred_Bbox = pred_box[...,self.C:self.B*4]
        true_Bbox = true_box[...,self.C:self.B*4]
        xy_p = pred_Bbox[...,:2]
        xy_t = true_Bbox[...,:2]
        hw_p = pred_Bbox[...,2:]
        hw_t = true_Bbox[...,2:]

        xyloss = tf.nn.l2_loss(xy_p-xy_t)
        hwloss = tf.nn.l2_loss(tf.math.sqrt(hw_p) -tf.math.sqrt(hw_t))
        return self.coord_scale * (xyloss+hwloss)
    def object_loss(self, pred_box, true_box):
        
        return self.obj_scale * tf.nn.l2_loss(pred_box - true_box)
    def noobject_loss(self, pred_box, true_box):

        return self.noobj_scale * tf.nn.l2_loss(pred_box - true_box)
    def class_loss(self, pred_box, true_box):
        pred_box = true_box * pred_box

        return self.class_scale * tf.nn.l2_loss(pred_box - true_box)
    def calc_iou(self, boxes1, boxes2):
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

    def call(self, y_true, y_pred):
        #讓network output的shape和label一樣
        y_pred = tf.reshape(y_pred, self.outputsize) # 7,7,30

        #區分有無物件
        exist_object_mask = y_true[...,-1] #最後一欄是conf
        exist_object_mask = tf.tile(tf.expand_dims(exist_object_mask, axis=-1), [1,1,1,30])

        #為了計算iou提出BOX的部分
        pred_box = y_pred[...,self.C:self.C+4*self.B] #7,7,8
        pred_box = tf.reshape(pred_box, (-1,4))

        true_box = y_true[...,self.C:self.C+4*self.B] #7,7,8
        true_box = tf.reshape(true_box, (-1,4)) 

        #計算所有cell上的box的iou
        allboxiou = self.calc_iou(true_box, pred_box)
        allboxiou = tf.reshape(allboxiou, (-1,2))
        allboxiou = tf.reshape(allboxiou, (-1,7,7,2))
        #挑出同cell中iou較大的box
        iou_argmax = tf.math.argmax(allboxiou, axis=3)
        iou_argmax = tf.reshape(iou_argmax, (-1,7,7,1))        
        not_ioubox = tf.cast(tf.math.logical_not(tf.cast(iou_argmax,tf.bool)),tf.float32)
        ioubox = tf.cast(iou_argmax,tf.float32)
        dummy_class = tf.ones([self.batch_size, self.S, self.S, self.C])
        #產生一個mask遮掩住iou較小的部分
        iou_responsible_box_mask = tf.concat([dummy_class, \
                                              tf.tile(not_ioubox,[1,1,1,4]), \
                                              tf.tile(ioubox,[1,1,1,4]), \
                                              not_ioubox, \
                                              ioubox], 3)
        object_masked_y_true = exist_object_mask * iou_responsible_box_mask * y_true
        object_masked_y_pred = exist_object_mask * iou_responsible_box_mask * y_pred

        coord_loss = self.coordinate_loss(object_masked_y_pred, object_masked_y_true)
        class_loss = self.class_loss(object_masked_y_pred[...,:self.C], object_masked_y_true[...,:self.C])

        best_pre_ious = exist_object_mask[...,-2:] * iou_responsible_box_mask[...,-2:] * allboxiou
        best_true_box = exist_object_mask[...,-2:] * iou_responsible_box_mask[...,-2:] * y_true[...,-2:]

        object_loss = self.object_loss(best_pre_ious, best_true_box)

        no_object_mask = tf.math.abs((1 - exist_object_mask))
        noobject_box = no_object_mask[...,-2:] * y_pred[...,-2:]
        noobject_true_box = no_object_mask[...,-2:] * y_true[...,-2:]

        noobject_loss = self.noobject_loss(noobject_box, noobject_true_box)

        return (coord_loss + object_loss + noobject_loss + class_loss)/self.batch_size

def test():
    a = tf.random.normal([16,7,7,30]).numpy()
    a[0,3,6,-1]=1
    a[0,3,5,-1]=1
    a = tf.constant(a) 
    b = tf.random.normal([16,7,7,30])
    yololoss = yolov1_loss()
    #d = yololoss.darknet19()
    #d.summary()
    loss = yololoss.call(a,b)

def train_test():
    yolo_model =yolov1(7,2,20)
    dataset_path = 'E:\\dataset\\pascal2012'
    batch_size = 4
    train_gen = pascal_dataloader.pascal_dataloader(dataset_path, batch_size, mode=0)
    test_gen = pascal_dataloader.pascal_dataloader(dataset_path, batch_size, mode=1)
    
    yolo_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4), loss=yolov1_loss(batch_size))

    callbacks = [tf.keras.callbacks.ModelCheckpoint('model/weights.h5', save_best_only=True, save_weights_only=True)]
    history = yolo_model.fit_generator(train_gen.generator(), steps_per_epoch=100, epochs=135, validation_data=test_gen.generator(), validation_steps=30,callbacks=callbacks)
if __name__ =='__main__':
    train_test()