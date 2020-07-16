import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

# suppress the signal share with the images from the same camera model
def Zeromean(X):
    M,N = X.shape[0],X.shape[1]
    Xzm = X - np.mean(X)
    MeanR = [np.mean(X,0)]
    MeanC = [np.mean(X,1)]

    OneCol = np.ones_like(np.transpose(MeanC))
    LPR = OneCol @ MeanR
    # plt.imshow(LPR,cmap="gray",vmin=-1, vmax=1)
    # plt.colorbar()
    # plt.show()

    OneRow = np.ones_like(MeanR)
    LPC = np.transpose(MeanC) @ OneRow
    # plt.imshow(LPC,cmap="gray",vmin=-1, vmax=1)
    # plt.colorbar()
    # plt.show()

    ZMX = Xzm-LPR-LPC

    return ZMX
