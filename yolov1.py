import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer

class yolo(Model):
    def __init__(self):
        super(yolo,self).__init__()
        self.imageSize = (448, 448, 3)
        self.num_classes = 20
        self.S = 7
        self.B = 2
        self.batch_size = 16
        #self.weight_decay = 0.01
        self.leaky_relu_alhpa = 0.1
        self.obj_scale = 1
        self.noobj_scale = 0.5
        self.coord_scale = 5
        self.class_scale = 1

        self.conf_threshold = .5
        self.iou_threshold = .5

        #create model
        model = tf.keras.Sequential()
        model.add(Conv2D(64,7,2,padding='same',activation=tf.nn.leaky_relu(alpha=self.leaky_relu_alhpa)))
        model.add(MaxPool2D(2,2,padding='same'))

        model.add(Conv2D(192,3,padding='same',activation=tf.nn.leaky_relu(alpha=self.leaky_relu_alhpa)))
        model.add(MaxPool2D(2,2,padding='same'))

        model.add(Conv2D(128,1,padding='same',activation=tf.nn.leaky_relu(alpha=self.leaky_relu_alhpa)))
        model.add(Conv2D(256,3,padding='same',activation=tf.nn.leaky_relu(alpha=self.leaky_relu_alhpa)))
        model.add(Conv2D(256,1,padding='same',activation=tf.nn.leaky_relu(alpha=self.leaky_relu_alhpa)))
        model.add(Conv2D(512,3,padding='same',activation=tf.nn.leaky_relu(alpha=self.leaky_relu_alhpa)))
        model.add(MaxPool2D(2,2,padding='same'))

        for i in range(3):
            model.add(Conv2D(256,1,padding='same',activation=tf.nn.leaky_relu(alpha=self.leaky_relu_alhpa)))
            model.add(Conv2D(512,3,padding='same',activation=tf.nn.leaky_relu(alpha=self.leaky_relu_alhpa)))
        model.add(Conv2D(512,1,padding='same',activation=tf.nn.leaky_relu(alpha=self.leaky_relu_alhpa)))
        model.add(Conv2D(1024,3,padding='same',activation=tf.nn.leaky_relu(alpha=self.leaky_relu_alhpa)))
        model.add(MaxPool2D(2,2,padding='same'))

        for i in range(1):
            model.add(Conv2D(512,1,padding='same',activation=tf.nn.leaky_relu(alpha=self.leaky_relu_alhpa)))
            model.add(Conv2D(1024,3,padding='same',activation=tf.nn.leaky_relu(alpha=self.leaky_relu_alhpa)))
        model.add(Conv2D(1024,3,padding='same',activation=tf.nn.leaky_relu(alpha=self.leaky_relu_alhpa)))
        model.add(Conv2D(1024,3,2,padding='same',activation=tf.nn.leaky_relu(alpha=self.leaky_relu_alhpa)))
        model.add(Conv2D(1024,3,padding='same',activation=tf.nn.leaky_relu(alpha=self.leaky_relu_alhpa)))
        model.add(Conv2D(1024,3,padding='same',activation=tf.nn.leaky_relu(alpha=self.leaky_relu_alhpa)'))

        model.add(Flatten())
        model.add(Dense(4096,activation=tf.nn.leaky_relu(alpha=self.leaky_relu_alhpa)))
        #output shape = S * S * (Box數量 * 5 + 類別數量)
        model.add(Dense(self.S*self.S*(self.B*5+self.num_classes),activation='linear'))
        
        self.model = model
    
    def call(self,input):
        x = tf.keras.Input(shape=(448,448,3))
        return self.model(x)

    def conv2d(self,input,filters,slide,set_batch_norm=False):
        
        
    def yolo_loss(self, netoutput, GT):
        #netoutput [batch, S*S*(2*B+CLASS)]
        #ground truth [batch, S*S*25]
        tf.reshape(netoutput, [self.S,self.S,2*self.B+self.classes])
        

m = yolo()
m.build(input_shape=[None,448,448,3])
m.summary()
