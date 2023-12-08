import tensorflow as tf
from pascal_dataloader import PascalDataLoader
from yolo import yolov1, yolov1_head, yolov1_loss
from callbacks import Stabilizer
import math
# 查看可用的 GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # 將 GPU 設定為僅在需要時申請內存
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("可用的 GPU 數量：", len(gpus))
else:
    print("沒有可用的 GPU。將使用 CPU。")


dataset_path = 'D:\\programming\\MLDL\\dataset\\pascal2012'
batch_size = 8
train_gen = PascalDataLoader(dataset_path, batch_size, mode=0)
#val_gen = PascalDataLoader(dataset_path, batch_size, mode=1)
dataset = train_gen.create_dataset()
train_dataset = dataset.repeat()
initial_learning_rate =1e-4
inputs_ts = tf.keras.Input([224,224,3])
outputs_ts = yolov1(S=7,B=2,C=20)(inputs_ts)
yolo_model = tf.keras.Model(inputs=inputs_ts, outputs=outputs_ts)
yolo_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate), loss=yolov1_loss(batch_size,B=2))
history = yolo_model.fit(train_dataset,
                        batch_size=batch_size,
                        epochs=(135),
                        steps_per_epoch=math.ceil(train_gen.datasetLen/batch_size),
                        callbacks=[Stabilizer()])
yolo_model.save_weights('model/full_train1208.hdf5')