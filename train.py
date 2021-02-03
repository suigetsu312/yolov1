from model import yolo
import tensorflow as tf
from pascal_dataloader import pascal_dataloader

callbacks = [tf.keras.callbacks.ModelCheckpoint(
    'model/weights.h5', save_best_only=True, save_weights_only=True)]
model = yolo()
train_steps = 10
dataset_path = 'E:\\dataset\\pascal2012'
train_gen = pascal_dataloader(dataset_path, 16)
model.compile(optimizer=tf.optimizers.Adam(learning_rate=5e-4), loss = model.yolo_loss)
model.fit_generator(train_gen.__next__(), steps_per_epoch=10, epochs=100, callbacks=callbacks)
