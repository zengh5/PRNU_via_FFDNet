import numpy as np
# we do not consider shift here
def PCE1(C):
    r = 5
    M,N= C.shape[0],C.shape[1]
    Cinrange = C[M-1:M, N-1:N]
    # Make the neighbor of the peak located at down-right corner
    temp = np.roll(C,-r,axis=0)
    Rollc = np.roll(temp, -r, axis=1)
    # set the the neighbor of the peak 0
    Rollc[M-2*r-1:M,N-2*r-1:N] = 0
    # the result should be the same as the Matlab version
    PCE_energy = np.sum(Rollc * Rollc)/ (M * N - np.power(2 * r + 1, 2))
    peakheight = C[M-1,N-1]
    # Square of the peakvalue with SIGN
    Height2 = np.sign(peakheight)* np.power(peakheight, 2)
    PCE = Height2/ PCE_energy
    Out = {'peakheight': peakheight,'PCE':PCE}
    return Out['PCE']
