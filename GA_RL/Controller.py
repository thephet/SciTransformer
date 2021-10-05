#########################################################################################
#
# Following "worlds model" (search for this paper) the controller will be a simple NN
# that receives as input the state of the transformer and the new input.
# And it outputs the next action.
#
#########################################################################################

import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys

# imports from parent folder
sys.path.append("..")
from Transformer import Transformer


class Controller():

    def __init__(self, t_weights, output_size, c_weights=None):

        # create an empty transformers, I cant just load the model
        self.t = Transformer(4,128,8,1024,25,25,300,300)
        # now load weights
        self.t.model.load_weights(t_weights)

        # the last layer that will output the motors
        # only training will happen here
        self.pre_output = keras.layers.Dense(
                1024, activation='relu', name='preoutput')
        self.output = keras.layers.Dense(
                output_size, activation='sigmoid')#, name='output')

        self._build()

        if c_weights:
            self.model.load_weights(c_weights)


    def _build(self):
        
        # First we need to calculate the output of the transformer's decoder
        trans_inp = keras.layers.Input(shape=(None, 25))#, name="t_in")
        trans_target = keras.layers.Input(shape=(None, 25))#, name="t_tar")
        enc_output = self.t.encoder(trans_inp)
        dec_output = self.t.decoder(trans_target, enc_output)
        # we dont want to train them
        self.t.model.trainable = False

        # return it through the dense layer
        pre_output = self.pre_output(dec_output)
        output = self.output(pre_output)
        output = tf.expand_dims( output[:,-1], axis=1 )

        # keras training model
        self.model = keras.models.Model( 
                inputs=[trans_inp, trans_target], 
                outputs=output)
        self.model.add_loss( self.custom_loss( 
            trans_inp, trans_target, output) )
        self.model.compile( loss=None, optimizer='adam' )
        #self.model.summary()
        
    
    def custom_loss(self, t_in, t_target, y_pred):
        '''data will contain inputs to encoder and decoder'''

        # add new motor prediction to trans_inp
        new_motors = keras.layers.Concatenate(axis=1)([t_in, y_pred])
        # and remove oldest one
        new_motors = new_motors[:,1:]

        # now predict using transformer, get [1] because [0] is reconstruction
        prediction = self.t.model( [ new_motors, t_target ] )[1]

        # now we can calculate value of center cell [12] against the others
        total_sum = keras.backend.sum(prediction, axis=2)
        center_sum = prediction[:,:,12]
        value = (center_sum * 25 - total_sum) / 25.
        # value now will be between -1 to 1, rescale to 0 to 1
        value = (value+1)/2.
        return tf.math.exp( value[:,-1] )
        # return tf.math.exp( 1 - value[:,-1] )


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
                # motor pattern, BZ oscillation. The output doesnt
                # matter because it's never used in the loss function
                yield ([x[...,:25], x[...,25:] ], y)


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
        filepath = folderpath+"cbinsigneg-{epoch:03d}-{loss:.4f}-{val_loss:.4f}.hdf5"    

        checkpoint1 = keras.callbacks.ModelCheckpoint(
            filepath, monitor='loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            period=5,
            mode='min')

        callbacks_list = [ checkpoint1, 
                keras.callbacks.ReduceLROnPlateau(
                    monitor='loss', patience=25, cooldown=25, verbose=1) ]
        
        self.model.fit(
                x = self.customGenerator(gen, randomize[:trainbatches]), 
                y = None, 
                validation_data = self.customGenerator(gen, randomize[trainbatches:]),
                epochs=1000, steps_per_epoch = trainbatches, 
                validation_steps = valbatches,
                callbacks=callbacks_list)

        return model
