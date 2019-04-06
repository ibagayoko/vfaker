import keras
import tensorflow as tf

class TempMasking(keras.layers.Layer):
    '''
    Create function for temporal masking. Use librosa.decompose.hpss.
    Split and concatinate dimensions to make it 2D.

    '''
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(TempMasking, self).__init__(**kwargs)
    def call(self, inputs):
      return inputs
  