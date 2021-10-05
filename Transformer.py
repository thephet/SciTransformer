#########################################################################################
#
# This code implements a vanilla Transformer as seen in "Attention is all you need"
# It has some modifications to adapt the model to work with scientific data. See the
# paper related to this code.
#
# To understand the following code, you first need to understand the Transformer arch.
# Do the following steps before trying to read this code:
#
# 1. Read the paper mentioned above, or check tutorials about attention, self attention
#   and transformers. Seqseq is also recommended.
# 2. Read the TF tutorial about Transformers: 
#   https://www.tensorflow.org/tutorials/text/transformer
#   I also suggest you check the other tutorials related to attention.
# 3. Check this website: 
#   https://medium.com/@max_garber/simple-keras-transformer-model-74724a83bb83
#
# My code is based on points 2/3 with the required variations required for this research.
#
# To simplify it, I will make sure of the sequences have the same length. So pad masking
# won't be required.
#
##########################################################################################


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ["CUDA_VISIBLE_DEVICES"] = '6'

import tensorflow as tf
import numpy as np


# POSITIONAL ENCODING FUNCTIONS
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles( np.arange(position)[:, np.newaxis], 
            np.arange(d_model)[np.newaxis, :], d_model)
    
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)



# Initially I was subclassing from keras model, but doing so I couldnt use some of the
# stuff from Keras such as fit, and instead I had to train the epochs manually.
# So I will create a regular Python class which has a keras model as attribute.
class Transformer():

    def __init__(self, num_layers, d_model, num_heads, dff, input_size, target_size, 
            pe_input, pe_target, do_rate=0.1):

        self.d_model = d_model # need to save this one for CustomSchedule 
        self.encoder = Encoder(input_size, num_layers = num_layers, d_model = d_model, 
                num_heads = num_heads, max_pe = pe_input, dff = dff, dropout = do_rate)
        self.decoder = Decoder(target_size, num_layers = num_layers, d_model = d_model, 
                num_heads = num_heads, max_pe = pe_target, dff = dff, dropout = do_rate)

        # input encoding reconstruction
        self.re_prev_layer = tf.keras.layers.Dense(dff)
        #self.re_final_layer = tf.keras.layers.Dense(target_size, activation='relu')
        self.re_final_layer = tf.keras.layers.Dense(target_size, activation='elu')

        # transformer decoding output
        self.prev_layer = tf.keras.layers.Dense(dff)
        # for binary outputs - use this or the next one. not both
        # self.final_layer = tf.keras.layers.Dense(target_size, activation='sigmoid')
        # for raw outputs (it predicts the blue channel)
        self.final_layer = tf.keras.layers.Dense(target_size, activation='relu')

        self._build()

    def _build(self):

        inp = tf.keras.layers.Input(shape=(None, 25))
        target = tf.keras.layers.Input(shape=(None, 25))

        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inp)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output = self.decoder(target, enc_output)
        # (batch_size, tar_seq_len, target_vocab_size)
        prev_layer = self.prev_layer(dec_output)
        final_output = self.final_layer(prev_layer)

        # input reconstruction
        r_prev_layer = self.re_prev_layer(enc_output)
        recons_output = self.re_final_layer(r_prev_layer)

        self.model = tf.keras.models.Model(inputs=[inp, target], 
                outputs=[recons_output, final_output])
        self.enc_model = tf.keras.models.Model(inputs=inp, outputs=recons_output)



class Encoder(tf.keras.layers.Layer):
    
    def __init__(self, input_size, num_layers = 4, d_model = 512, num_heads = 8, 
            dff = 2048, max_pe = 10000, dropout = 0.0):
        super(Encoder, self).__init__()

        self.input_size, self.num_layers, self.d_model = input_size, num_layers, d_model
        self.num_heads, self.dff, self.max_pe, self.dropout = num_heads, dff, max_pe, dropout

        #self.embedding = tf.keras.layers.Embedding(input_size, d_model, mask_zero=True)
        self.embedding = tf.keras.layers.Dense(d_model)
        self.pos = positional_encoding(max_pe, d_model)

        self.encoder_layers = [ EncoderLayer(d_model = d_model, num_heads = num_heads, 
            dff = dff, dropout = dropout) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout)


    def call(self, inputs, mask=None, training=None):
        seq_len = tf.shape(inputs)[1] # length of the input sequence

        # adding embedding and position encoding.
        x = self.embedding(inputs) # (batch_size, input_seq_len, d_model)
        # positional encoding
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) 
        x += self.pos[: , :seq_len, :]

        x = self.dropout(x, training=training)

        #Encoder layer
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask = mask)

        return x # (batch_size, input_seq_len, d_model)



class Decoder(tf.keras.layers.Layer):

    def __init__(self, target_size, num_layers = 4, d_model = 512, num_heads = 8, 
          dff = 2048, max_pe = 10000, dropout = 0.0):
        super(Decoder, self).__init__()

        self.target_size, self.num_layers, self.d_model = target_size, num_layers, d_model
        self.num_heads, self.dff, self.max_pe, self.dropout = num_heads, dff, max_pe, dropout

        #self.embedding = tf.keras.layers.Embedding(target_size, d_model, mask_zero=True)
        self.embedding = tf.keras.layers.Dense(d_model)
        self.pos = positional_encoding(max_pe, d_model)

        self.dec_layers = [ DecoderLayer(d_model = d_model, num_heads = num_heads, 
            dff = dff, dropout = dropout)  for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout)


    def call(self, x, enc_output, mask=None, training=None):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x) # (batch_size, target_seq_len, d_model)
        # positional encoding
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos[: , :seq_len, :]

        x = self.dropout(x, training=training)

        #Decoder layer
        for decoder_layer in self.dec_layers:
            x = decoder_layer(x, enc_output, mask = mask)

        return x



class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self,  d_model = 512, num_heads = 8, dff = 2048, dropout = 0.0):
        super(EncoderLayer, self).__init__()

        self.d_model, self.num_heads, self.dff, self.dropout = d_model, num_heads, dff, dropout

        self.mha =  MultiHeadAttention(d_model, num_heads)
        self.dropout_attention = tf.keras.layers.Dropout(dropout)
        self.add_attention = tf.keras.layers.Add()
        self.layer_norm_attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
        self.dense2 = tf.keras.layers.Dense(d_model)
        self.dropout_dense = tf.keras.layers.Dropout(dropout)
        self.add_dense = tf.keras.layers.Add()
        self.layer_norm_dense = tf.keras.layers.LayerNormalization(epsilon=1e-6)


    def call(self, inputs, mask=None, training=None):
        attention = self.mha([inputs,inputs,inputs], mask = [mask,mask])
        attention = self.dropout_attention(attention, training = training)
        x = self.add_attention([inputs , attention])
        x = self.layer_norm_attention(x)

        ## Feed Forward
        dense = self.dense1(x)
        dense = self.dense2(dense)
        dense = self.dropout_dense(dense, training = training)
        x = self.add_dense([x , dense])
        x = self.layer_norm_dense(x)

        return x



class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self,  d_model = 512, num_heads = 8, dff = 2048, dropout = 0.0):
        super(DecoderLayer, self).__init__()

        self.d_model, self.num_heads, self.dff, self.dropout = d_model, num_heads, dff, dropout

        self.mha1 =  MultiHeadAttention(d_model, num_heads, causal = True)
        self.dropout_attention1 = tf.keras.layers.Dropout(dropout)
        self.add_attention1 = tf.keras.layers.Add()
        self.layer_norm_attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.mha2 =  MultiHeadAttention(d_model, num_heads)
        self.dropout_attention2 = tf.keras.layers.Dropout(dropout)
        self.add_attention2 = tf.keras.layers.Add()
        self.layer_norm_attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)


        self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
        self.dense2 = tf.keras.layers.Dense(d_model)
        self.dropout_dense = tf.keras.layers.Dropout(dropout)
        self.add_dense = tf.keras.layers.Add()
        self.layer_norm_dense = tf.keras.layers.LayerNormalization(epsilon=1e-6)


    def call(self, x, enc_output, mask=None, training=None):
        attention = self.mha1([x,x,x], mask = mask)
        attention = self.dropout_attention1(attention, training = training)
        x = self.add_attention1([x , attention])
        x = self.layer_norm_attention1(x)
        
        attention = self.mha2([x, enc_output, enc_output], mask = mask)
        attention = self.dropout_attention2(attention, training = training)
        x = self.add_attention1([x , attention])
        x = self.layer_norm_attention1(x)

        ## Feed Forward
        dense = self.dense1(x)
        dense = self.dense2(dense)
        dense = self.dropout_dense(dense, training = training)
        x = self.add_dense([x , dense])
        x = self.layer_norm_dense(x)

        return x


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model = 512, num_heads = 8, causal=False, dropout=0.0):
    super(MultiHeadAttention, self).__init__()

    assert d_model % num_heads == 0
    depth = d_model // num_heads

    self.w_query = tf.keras.layers.Dense(d_model)
    self.split_reshape_query = tf.keras.layers.Reshape((-1,num_heads,depth))  
    self.split_permute_query = tf.keras.layers.Permute((2,1,3))      

    self.w_value = tf.keras.layers.Dense(d_model)
    self.split_reshape_value = tf.keras.layers.Reshape((-1,num_heads,depth))
    self.split_permute_value = tf.keras.layers.Permute((2,1,3))

    self.w_key = tf.keras.layers.Dense(d_model)
    self.split_reshape_key = tf.keras.layers.Reshape((-1,num_heads,depth))
    self.split_permute_key = tf.keras.layers.Permute((2,1,3))

    self.attention = tf.keras.layers.Attention(causal=causal, dropout=dropout)
    self.join_permute_attention = tf.keras.layers.Permute((2,1,3))
    self.join_reshape_attention = tf.keras.layers.Reshape((-1,d_model))

    self.dense = tf.keras.layers.Dense(d_model)

  def call(self, inputs, mask=None, training=None):
    q = inputs[0]
    v = inputs[1]
    k = inputs[2] if len(inputs) > 2 else v

    query = self.w_query(q)
    query = self.split_reshape_query(query)    
    query = self.split_permute_query(query)                 

    value = self.w_value(v)
    value = self.split_reshape_value(value)
    value = self.split_permute_value(value)

    key = self.w_key(k)
    key = self.split_reshape_key(key)
    key = self.split_permute_key(key)

    if mask is not None:
      if mask[0] is not None:
        mask[0] = tf.keras.layers.Reshape((-1,1))(mask[0])
        mask[0] = tf.keras.layers.Permute((2,1))(mask[0])
      if mask[1] is not None:
        mask[1] = tf.keras.layers.Reshape((-1,1))(mask[1])
        mask[1] = tf.keras.layers.Permute((2,1))(mask[1])

    attention = self.attention([query, value, key], mask=mask)
    attention = self.join_permute_attention(attention)
    attention = self.join_reshape_attention(attention)

    x = self.dense(attention)

    return x
    

if __name__ == "__main__":
    # num layers, dsize, heads, dff, inputsize, outputsize, max input, max output
    t = Transformer(4, 25, 5, 512, 25, 25, 240, 240)
