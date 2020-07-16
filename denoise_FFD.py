
import numpy as np
import torch
from torch.autograd import Variable
from utils import normalize, init_logger_ipol


def denoise_FFD(imorig,noise_sigma,cuda,model):
    # logger = init_logger_ipol()

    imorig = np.expand_dims(imorig, 0)
    imorig = normalize(imorig)
    imorig = torch.Tensor(imorig)

    # Load saved weights
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    # model.load_state_dict(state_dict)
    # model.load_state_dict(state_dict['state_dict'])
    # Sets the model in evaluation mode (e.g. it removes BN)
    # model.eval()
    # we do not need to add noise
    imnoisy = imorig.clone()

    # Test mode
    with torch.no_grad():  # PyTorch v0.4.0
        imorig, imnoisy = Variable(imorig.type(dtype)), Variable(imnoisy.type(dtype))
        nsigma = Variable(torch.FloatTensor([noise_sigma]).type(dtype))

    # Estimate noise and subtract it to the input image
    im_noise_estim = model(imnoisy, nsigma)
    outim = torch.clamp(imnoisy-im_noise_estim, 0., 1.)

    # return outim
    imnoise_es_cpu = (im_noise_estim.data.cpu().numpy()[0, 0, :]*255.).astype(np.float32)
    imnoise_es_cpuclip = ((imnoisy-outim).data.cpu().numpy()[0, 0, :] * 255.).astype(np.float32)
    # return variable_to_cv2_image(outim)
    return imnoise_es_cpuclip