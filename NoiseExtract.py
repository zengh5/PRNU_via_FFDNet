# pywavelets package is needed
# Implemented by Hui Zeng when visiting in Binghamton University
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import pywt

def NoiseExtract(img, qmf, sigma, L):
    L = 4
    height, width = img.shape[0], img.shape[1]
    img = img.astype(np.float32)
    # make sure the width and the height are Divisible by  2^L , padding.
    m = np.power(2,L)
    nr = math.ceil(height / m) * m
    nc = math.ceil(width / m) * m
    padr = ((nr-height)/2).astype(np.int)
    padc = ((nc-width)/2).astype(np.int)
    img_pad = np.pad(img, ((padr, padr), (padc, padc)), 'symmetric')
    # plt.figure(figsize=(3, 3))
    # plt.imshow(np.array(img_pad,dtype='uint8'), cmap="gray", vmin=0, vmax=255)
    # plt.show()
    NoiseVar =  np.power(sigma,2)
# the result is the same as wavedec2 in Matlab.
# slightly different with mdwt in the direction of filter and shift.
# The denoising performance should be the same as mdwt

    # wavelet decomposition
    wave_trans = pywt.wavedec2(img_pad, 'db4', level=L, mode='periodization')

    LL = wave_trans[0]
    # clear the LL band
    wave_trans_dn=[np.zeros_like(LL, dtype=np.float32)]
    for i in range(1, L + 1):
        HH = wave_trans[i][2]
        LH = wave_trans[i][1]
        HL = wave_trans[i][0]
        # The core of Mihçak's method, [8]
        temp3 = WaveNoise(HH, NoiseVar)
        temp2 = WaveNoise(LH, NoiseVar)
        temp1 = WaveNoise(HL, NoiseVar)
        t = (temp1, temp2, temp3)
        wave_trans_dn.append(t)

    rec_im = pywt.waverec2(wave_trans_dn, 'db4', mode='periodization')
    image_noise = rec_im[padr:(padr + height), padc: (padc + width)]
    return image_noise

# Mihçak's method, [8]：'We assume that the variance field is smoothly changing.'
# This genius idea was first proposed in 'SPATIALLY ADAPTIVE STATISTICAL MODELING OF WAVELET IMAGE
# COEFFICIENTS AND ITS APPLICATION TO DENOISING', ICASSP1999. However, the implementation is indeed according to
# 'Low-Complexity Image Denoising Based on Statistical Modeling of Wavelet Coefficients' IEEE SIGNAL PROCESSING LETTERS
# 1999
def WaveNoise(coef,NoiseVar):
    # the expectation of wavelet coefficients is approximately zero，D(X) = E(X^2)-(E(X)^2) =E(X^2)
    # So, EstVar1 is the local variance
    tc = coef*coef
    # cv2.blur() local average
    EstVar1 = cv2.blur(tc, (3, 3))
    # Soft threshold shrink
    temp = EstVar1 - NoiseVar
    coefVar = np.maximum(temp,0)
    # For robustness, windows of different sizes are used. The one who has the minimum variance is the winner
    for w in range(5, 10, 2):
        EstVar1 = cv2.blur(tc, (w, w))
        temp = EstVar1- NoiseVar
        EstVar = np.maximum(temp,0)
        coefVar = np.minimum(coefVar, EstVar)

    # coefVar=0, all noise, tc =coef
    # coefVar>>NoiseVar, all textures, tc = 0
    tc = (coef* NoiseVar)/ (coefVar + NoiseVar)
    return tc
    # return wave_im
    # wave_trans = mdwt(img_pad, qmf, L);