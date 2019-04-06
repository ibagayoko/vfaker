import keras
import tensorflow as tf
import math

# Based on a pytorch implementation from
# [Conv1DGLU pytorch](https://github.com/Sharad24/Neural-Voice-Cloning-with-Few-Samples/blob/master/Modules/Conv1dGLU.py)
class Conv1DGLU(keras.layers.Layer):
    '''
        Implementation of the Conv1d + GLU(Gated Linear Unit)
        with residual connection.
        For GLU refer to https://arxiv.org/abs/1612.08083 paper.
        '''
    def __init__(self, output_dim, filters = 128, kernel_size=12, padding = None, dilation = 2, **kwargs):
        self.output_dim = output_dim
        super(Conv1DGLU, self).__init__(**kwargs)
   
        if padding == None:
            self.padding = int(((kernel_size-1)/2)*dilation)
        self.conv1 = keras.layers.Conv1D(filters, kernel_size, dilation_rate=dilation)
    def call(self, inputs):
      residual = inputs
      
      # convolution 1d
      c1 = self.conv1(inputs)
      c2 = self.conv1(inputs)

      c1 = keras.layers.ZeroPadding1D(padding=self.padding)(c1)
      c2 = keras.layers.ZeroPadding1D(padding=self.padding)(c2)
      
    #   print(c1.shape)
      x = tf.multiply(c1, tf.sigmoid(c2, name="sigmoid"))
      
      # parties residual
      x = keras.layers.add([x, residual])
      x *= math.sqrt(0.5)
      
      return x
      