import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
# Compute the correlation between two matrices with FFT

def crosscorr(array1, array2):
    array1 = array1 - np.mean(array1)
    array2 = array2 - np.mean(array2)
    tilted_array2 = np.flipud(np.fliplr(array2))
    TA = np.fft.fft2(tilted_array2)
    FA = np.fft.fft2(array1)
    FF = FA * TA
    ret = np.real(np.fft.ifft2(FF))

    return ret.astype('float32')
