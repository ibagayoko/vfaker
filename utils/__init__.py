import numpy as np


def _add_pad(voices, maximum_size):
    '''Input:
    Specs: Mel Spectrograms with 80 channels but the length of each channel is not the same.
    maximum_size: Largest channel length. Others are padded to this length

    Padding with 0 won't affect the convolutions because anyway the neurons corresponding to the states have to
    be dead if they are not padded. Putting 0 will also make those neurons dead. And later an average is taken along
    this dimension too.

    Returns: A padded array of arrays of spectrograms.'''

    for i, i_element in enumerate(voices):
        for j, j_element in enumerate(i_element):
            final = np.zeros((maximum_size, 80))
            final[:voices[i][j].shape[0], :] += j_element
            #print(final.shape)
            voices[i][j]=final
    voices = np.array(voices)
    # print(voices.shape)
    return voices
def _largest_size(mels):
    """
    Find the largest size of the mel inputs
    """
    temp = [spec.shape[0] for text in mels for spec in text]
    largest_size = np.amax(np.array(temp))
    return largest_size

def prepareMelSpec(voices):
    """
    Expand all voices (mel specs)With the largest size 
    """
    largest_size = _largest_size(voices)
    return _add_pad(voices, largest_size)
  
