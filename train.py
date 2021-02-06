from model import yolo
import tensorflow as tf
from pascal_dataloader import pascal_dataloader

callbacks = [tf.keras.callbacks.ModelCheckpoint(
    'model/weights.h5', save_best_only=True, save_weights_only=True)]
yolo = yolo()
model = yolo.darknet19()
train_steps = 10
dataset_path = 'E:\\dataset\\pascal2012'
train_gen = pascal_dataloader(dataset_path, 16)
model.compile(optimizer=tf.optimizers.Adam(learning_rate=5e-4), loss = yolo.yolo_loss, metrics=[yolo.yolo_loss])
model.fit_generator(train_gen.generator(), steps_per_epoch= 1, epochs = 200, verbose=1, callbacks= callbacks)
