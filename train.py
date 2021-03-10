from model import yolo
import tensorflow as tf
from pascal_dataloader import pascal_dataloader

yolo_model =yolov1(7,1,20)
dataset_path = 'E:\\dataset\\pascal2012'
batch_size = 8
train_gen = pascal_dataloader.pascal_dataloader(dataset_path, batch_size, mode=0)

initial_learning_rate =1e-4
inputs_ts = tf.keras.Input([224,224,3])
outputs_ts = yolov1(S=7,B=1,C=20)(inputs_ts)
yolo_model = tf.keras.Model(inputs=inputs_ts, outputs=outputs_ts)
yolo_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate), loss=yolov1_loss(batch_size,B=1))
history = yolo_model.fit(train_gen.generator(), batch_size=batch_size, epochs=100, steps_per_epoch=10)

yolo_model.save_weights('model/full_train.hdf5')