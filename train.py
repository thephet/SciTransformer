#########################################################################################
#
# This will load a transformer model, using transformer.py, and either prepare the data
# with prepare_data.py or fetch the already prepared pickle file.
# And then compile the model and train it, using the usual Keras functions.
#
# To deeply understand the following code, please check tutorials discussed
# in the readme.md. Following code is just an adaption of those
#
#########################################################################################


import tensorflow as tf
import numpy as np
import pickle, os
from datetime import datetime

# my own class
import Transformer
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=5000):
        super(CustomSchedule, self).__init__()

        self.d_modelnp = d_model
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class TrainModel():

    def __init__(self, transformer, data_generator):

        self.trans = transformer
        self.data_generator = data_generator


    def compile(self):
        # Original code uses a masked_loss to ignore the padding (entries as "0")
        # In our case there's no padding because all the sequences have the same length

        d_model = self.trans.d_model
        optimizer = tf.keras.optimizers.Adam(CustomSchedule(d_model), 
                beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        # Binary cross entropy used for binarized data with sigmoid at the end
        #loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction='none')
        
        # loss of the decoder
        loss = tf.keras.losses.MeanSquaredError(reduction='none')
        # loss of the reconstruction of the encoder
        loss_r = tf.keras.losses.MeanSquaredError()#reduction='none')

        def masked_loss(y_true, y_pred):
            # calculate number of experiments with at least one oscillation
            mask = tf.keras.backend.max(y_true, axis=2)
            # sum all the experiments with 1s (if there was a 1 there was an oscillation)
            ones = tf.reduce_sum(mask)
    
            # if in this experiment all the entries were 0, ie no oscillations
            if ones == 0:
                return loss_r(y_true, y_pred)

            # otherwise only consider entries with oscillations
            _loss = loss(y_true, y_pred)
            mask = tf.cast(mask, dtype=_loss.dtype)
            _loss *= mask

            return tf.reduce_sum(_loss)/tf.reduce_sum(mask)

        metrics = [loss, masked_loss]#, tf.keras.metrics.BinaryAccuracy()]
        self.trans.model.compile(optimizer=optimizer, 
                loss = [loss_r, masked_loss], metrics = metrics) 


    def encoder_compile(self):
        # Compiles ONLY the encoder that will try to do reconstruction

        d_model = self.trans.d_model
        optimizer = tf.keras.optimizers.Adam(CustomSchedule(d_model), 
                beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        loss = tf.keras.losses.MeanSquaredError()
        self.trans.enc_model.compile(optimizer=optimizer, loss = loss) 


    def customGenerator(self, generator, indexes):
        ''' generator must be a TimeseriesGenerator.
        indexes are the train indexes that will be used to create the batches
        indexes cant be bigger than the number of samples.
        It is suggested that train indexes are 90% and test indexes are 10%'''

        while True:
            np.random.shuffle(indexes)
            for i in indexes:
                x,y = generator[i]
                x = np.squeeze(x)
                # motor pattern, BZ oscillation, - motor pattern BZ oscillation t+1
                yield ([x[...,:25], x[...,25:]], [x[...,:25], y])


    def train(self, gen):
        # gen is a timeseries generator as provided by keras

        num_batches = len(gen)
        # generator random list of batches
        randomize = np.arange( num_batches - 1 ) # remove last post we filled with 0s
        np.random.shuffle(randomize)
        # set train and test set as 90/10
        trainbatches = int( 0.9*len(randomize) )
        valbatches = int( 0.1*len(randomize) )

        folderpath = "/home/juanma/data/BZ/transformer/"
        filepath = folderpath+"weights1-{epoch:03d}-{loss:.4f}-{val_loss:.4f}.hdf5"    

        checkpoint1 = tf.keras.callbacks.ModelCheckpoint(
            filepath, monitor='loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            period=5,
            mode='min'
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss'
            , restore_best_weights=True
            , patience = 100
        )

        callbacks_list = [ checkpoint1, early_stopping ]

        # save validation set for testing later on
        date = datetime.now().strftime('%Y-%m-%d')
        val_file = "val_"+date+".p"
        val_entries = randomize[trainbatches:]
        with open(val_file, 'wb') as f:
            pickle.dump( val_entries, f )
       
        # compile encoder and the whole model
        self.encoder_compile()
        self.compile()

        for i in range(10):
            # First train the encoder a bit for input reconstruction
            self.trans.enc_model.fit(
                    x = self.customGenerator(gen, randomize[:trainbatches]), 
                    y = None, 
                    validation_data = self.customGenerator(gen, randomize[trainbatches:]), 
                    epochs=30-i*2, steps_per_epoch = trainbatches, 
                    validation_steps = valbatches)

            # now train whole thing
            self.trans.model.fit(
                    x = self.customGenerator(gen, randomize[:trainbatches]), 
                    y = None, 
                    validation_data = self.customGenerator(gen, randomize[trainbatches:]), 
                    epochs=100, steps_per_epoch = trainbatches, 
                    validation_steps = valbatches, callbacks=callbacks_list)

        return model

if __name__ == "__main__":

    t = Transformer.Transformer(4, 25, 5, 512, 25, 25, 240, 240)
    
    # load data generator, I will load from pickle otherwise use prepare_data.py
    with open('/home/data/juanma/BZ/data.pickle', 'rb') as handle:
        gen = pickle.load(handle)
    pass


