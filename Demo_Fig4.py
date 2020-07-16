import os
import cv2
import numpy as np
import time
import glob
import matplotlib.pyplot as plt
from scipy.io import savemat,loadmat
import torch
import torch.nn as nn
# FFDNet related
from NoiseExtractFFD import NoiseExtractFFD
from utils import remove_dataparallel_wrapper, normalize
from models import FFDNet
# Python implementation of the DDE codes
from crosscorr import crosscorr
from PCE1 import PCE1

print(torch.__version__)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
###  parameters ###
noise_sigma = 3/255
# Turn on GPU or not according to your environment
# cuda = True
cuda = False
# Only Grayscale model is provided here, the RGB model is even better.
rgb_den = False
B = 2048.
# B = 256.
########
in_ch = 1
# This model was trained on the Waterloo Exploration Database, sigma: [0,20], 50 epochs. see Fig.2 of the paper
# It cost about 12 hours in a GPU server (Nvidia RTX 2080Ti with 11GB graphic memory)
model_fn = 'mat/net-gray-v2.pth'
# Absolute path to model file
model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), model_fn)
# Create model
print('Loading model ...\n')
net = FFDNet(num_input_channels=in_ch)

# Load saved weights
if cuda:
    state_dict = torch.load(model_fn)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    # Sets data type according to CPU or GPU modes
    dtype = torch.cuda.FloatTensor
else:
    state_dict = torch.load(model_fn, map_location='cpu')
    # CPU mode: remove the DataParallel wrapper
    state_dict = remove_dataparallel_wrapper(state_dict)
    model = net
    dtype = torch.FloatTensor
model.load_state_dict(state_dict)
# If use the mid-saved model, uncomment the following code
# model.load_state_dict(state_dict['state_dict'])
# Sets the model in evaluation mode (e.g. it removes BN)
model.eval()
# Load the fingerprint generated with a set of flat images
data = loadmat('mat\\FlatFingerprint1.mat')
Fingerprint = data['Fingerprint']
[M, N] = [Fingerprint.shape[0], Fingerprint.shape[1]]
up = int(M / 2 - B / 2)
down = int(M / 2 + B / 2)
left = int(N / 2 - B / 2)
right = int(N / 2 + B / 2)
Fingerprint = Fingerprint[up:down, left:right]

# If you want save the PCE values, uncomment the corresponding codes
# matname = 'mat\\FFDNet' + str(Ci) + '.mat'
# Read the images and do camera identification
files = "Images\\*.JPG"
print(files)
filenames = glob.glob(files)
num_of_im = min(150, len(filenames))
PCE_FFD = np.zeros(num_of_im)
for i in range(0, num_of_im):
    print(i)
    imxname = filenames[i]
    # read the image and transfer to grayscale
    imx = cv2.imread(imxname, cv2.IMREAD_GRAYSCALE)

    # Crop the B by B pixels from the center
    [M, N] = [imx.shape[0], imx.shape[1]]
    up = int(M / 2 - B / 2)
    down = int(M / 2 + B / 2)
    left = int(N / 2 - B / 2)
    right = int(N / 2 + B / 2)
    imx_crop = imx[up:down, left:right]

    # Core function，SPN extraction， Noisex = I - F(I)
    # If you want to compare with the other methods, e.g., DWT，GIF, BM3D, change the following function
    Noisex = NoiseExtractFFD(imx_crop, noise_sigma, cuda, model)
    imx_crop = imx_crop.astype(float)
    KI = imx_crop * Fingerprint
    C = crosscorr(Noisex, KI)
    # We do not consider the shift between the probe image and the Fingerprint
    PCE_FFD[i] = PCE1(C)
    # Store the data
    # mdict = {'PCE_FFD': PCE_FFD}
    # savemat(matname, mdict)
#### The code is for the reproduction of our paper submitted to WIFS2020, by Hui Zeng ####