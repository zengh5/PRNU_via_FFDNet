import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from NoiseExtract import WaveNoise

# This process is important: suppress the signal share with the images from the same camera model.
# 1）Transfer SPN to Fourier domain,
# 2）suppress the bright point with WaveNoise()
# 3）Transfer back to the spatial domain
def WienerInDFT(X,sigma):
    M,N = X.shape[0],X.shape[1]
    F = np.fft.fft2(X)
    Fmag = np.abs(F / np.sqrt(M * N))

    # plt.imshow(Fmag,cmap="gray",vmin=-10, vmax=10)
    # plt.colorbar()
    # plt.show()

    NoiseVar = np.power(sigma,2)
    Fmag1 = WaveNoise(Fmag, NoiseVar)

    fzero = np.where(Fmag == 0)
    Fmag[fzero] = 1
    Fmag1[fzero] = 0
    F = F* Fmag1/ Fmag
    noiseclean = np.real(np.fft.ifft2(F))
    # plt.imshow(noiseclean,cmap="gray",vmin=-1, vmax=1)
    # plt.colorbar()
    # plt.show()

    return noiseclean.astype('float32')
