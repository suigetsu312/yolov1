import tensorflow as tf

class Stabilizer(tf.keras.callbacks.Callback):
    def __init__(self,security_boundary=0.1):
        super(Stabilizer,self).__init__()
        self._security_boundary=1+security_boundary
        self._last_loss=None
    def on_train_begin(self,logs={}):
        if(os.path.isfile("stabilizer.hdf5")==True):
            os.remove("stabilizer.hdf5")
        self.model.save_weights("stabilizer.hdf5")
    def on_train_end(self,logs={}):
        os.remove("stabilizer.hdf5")
    def on_epoch_end(self,epoch,logs={}):
        loss=logs.get('loss')
        if(math.isnan(loss)==True):
            for var in self.model.optimizer.variables():
                var.assign(tf.zeros_like(var))
            self.model.load_weights("stabilizer.hdf5")
        elif(self._last_loss==None or loss<self._last_loss*self._security_boundary):
            self.model.save_weights("stabilizer.hdf5")
            self._last_loss=loss
