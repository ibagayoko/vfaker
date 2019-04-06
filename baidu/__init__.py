import keras
import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Flatten
from .layers import Conv1DGLU, MultiHeadAttention, TempMasking



def build_model(batch_size=10, N_samples=23, Mel_dim=80, tframe = 201, N_prenet=1, N_conv=1, embs_shape=16):
    """
    Keras implementation for Baidu's
    [Neural Voice Cloning with a Few Samples](https://arxiv.org/pdf/1802.06006.pdf)
    """

    def _reshapeAndTranspose(inputs):
        """ 
            Reshape for GLU layer that apply a 1-D convolution
        """
        inputs = tf.reshape(inputs,  (batch_size*N_samples,tframe, int(x.shape[3])))
        inputs =  tf.transpose(inputs,   [1, 0,2])

    def _undoReshapeAndTranspose(inputs):
        """ 
            Reshape back after GLU layer that apply a 1-D convolution
        """
        inputs =  tf.transpose(inputs,   [1, 0, 2])
        inputs = tf.reshape(inputs,  (batch_size, N_samples,tframe, int(x.shape[2])))

    def GlobalMeanPooling(inputs):
        return tf.keras.backend.mean(inputs, axis=2)
    def MulAndEmbedding(inputs):
        x, residual_x = inputs
        # Softsign
        x = tf.keras.backend.softsign(x)
        # Normalization
        x = tf.math.l2_normalize(x, dim=1)
        # We expand and transpose to be able to apply the mutiplication operation
        x = tf.expand_dims(x, axis=2)
        x = tf.transpose(x, [0,2,1])
        # Mul operation
        x = tf.matmul( x , residual_x)
        # Squeeze to get the a propriete shape
        x = tf.squeeze(x)

        return x

    # We build the model describe in the paper 

    # Input layer 
    inp  = keras.layers.Input(shape=(N_samples,tframe, Mel_dim))

    # The Spectral processing part
    # The prenet
    x = Dense(128, activation='elu')(inp)
    # Add the N-1 left prenet
    for _ in range(N_prenet-1):
        x = Dense(128, activation='elu')(x)


    # Temporal processing part

    # We reshae for GLU Layer
    x = Lambda(_reshapeAndTranspose)(x)
    # The GLU Layer
    for _ in range(N_conv):
        x = Conv1DGLU(128, name='glulayer')(x)
    # We reshae for GLU Layer
    x = Lambda(_undoReshapeAndTranspose )(x)

    x = TempMasking(128)(x)

    # Global Mean Pooling along axis
    x = Lambda(GlobalMeanPooling)(x)

    # Cloning samples attention part

    # A fully-connected for residual mul after
    residual_x = Dense(512)(x)

    x = MultiHeadAttention(2, 128)([x,x, x])

    # A fully connected after attention layers
    x = Flatten()(x)
    x = Dense(23, activation='elu')(x)

    x = Lambda(MulAndEmbedding)([x, residual_x])

    x = Dense(embs_shape, activation='linear')(x)

    encoder = keras.models.Model(inp, x)

    return encoder
