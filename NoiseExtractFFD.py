import numpy as np
from denoise_FFD import denoise_FFD
# Python implementation of the DDE codes
from Zeromean import Zeromean
from WienerInDFT import WienerInDFT


def NoiseExtractFFD(imorig, noise_sigma, cuda, model):
    imorig = np.expand_dims(imorig, 0)
    Noisex = denoise_FFD(imorig, noise_sigma, cuda, model)
    # Noisex = NoiseExtractFromIm(imx[0],2)
    # Postprocessing, suppress the signal share with the images from the same camera model
    Noisex = Zeromean(Noisex)
    std = np.std(Noisex)
    Noisex = WienerInDFT(Noisex, std)

    return Noisex