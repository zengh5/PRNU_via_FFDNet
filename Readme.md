This code is to reproduce our work [1] submitted to WIFS2020  
[1] Kang Deng, Morteza Darvish Morshedi Hosseini, Anjie Peng, Hui Zeng, and Miroslav Goljan*,
'Extracting Sensor Pattern Noise for Camera Identification with FFDNet'.

This implementation is based on the public camera identification code of DDE; http://dde.binghamton.edu/download/camera_fingerprint/
and also the public code of FFDNet with Pytorch: https://doi.org/10.5201/ipol.2019.231
Matias Tassano, Julie Delon, and Thomas Veit, An Analysis and Implementation of the FFDNet Image Denoising Method,
Image Processing On Line, 9 (2019), pp. 1–25.
Please also cite the original paper.

# Runtime: 
  Python 3.6, Pytorch 1.5.0+cu101  
  pywavelets package is needed  
  Compatible with both GPU and CPU

# Usage:
  Run the 'Demo_Fig4.py' to see the camera identification results of the provided images.  
  The demo images are from the DID database: T. Gole and R. Bohme, "Dresden image database for benchmarking digital image forensics" in Proc. ACM Symp. Appl. Comput., 2010, pp. 1584–1590.  
  You can also change to your own reference fingerprint and images.

