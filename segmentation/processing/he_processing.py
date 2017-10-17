import numpy as np
from numpy.linalg import eig
import os
from glob import glob
from PIL import Image


# ---------------------------------------------------------------
# Stain Normalization
# ---------------------------------------------------------------


def stain_normalization(I, Io=240.0, beta=0.15, alpha=1.0,
            HERef=np.array([(0.5626, 0.2159), (0.7201, 0.8012),
                            (0.4062, 0.5581)]),
            maxCRef=np.array([1.9705, 1.0308]), return_he=True):

    M, N, D = I.shape

    if (D == 4):
        I = I[:, :, 0:3]
        D = 3

    # if I is not a float with values ranging 0-1, convert it
    if (I.dtype == 'uint8'):
        I = I.astype('float')

    I = np.reshape(I, (M * N, D), order='F')

    # calculate optical density
    OD = -np.log((I + 1) / Io)

    # remove transparent pixels
    ODhat = OD[~np.any(OD < beta, 1), :]

    # calculate eigenvectors
    W, V = eig(np.cov(ODhat.T))

    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    That = np.dot(ODhat, V[:, 0:2])

    # find the min and max vectors and project back to OD space
    phi = np.arctan2(That[:, 0], That[:, 1])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)

    vMin = np.dot(V[:, 0:2], np.array([(np.sin(minPhi)), (np.cos(minPhi))]))
    vMax = np.dot(V[:, 0:2], np.array([(np.sin(maxPhi)), (np.cos(maxPhi))]))

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array([vMin, vMax])
    else:
        HE = np.array([vMax, vMin])

    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (OD.shape[0], 3), 'F').T

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE.T, Y)
    C = C[0]

    # normalize stain concentrations
    maxC = np.percentile(C, 99, 1)

    C = C.T / maxC
    C = C * maxCRef
    C = C.T

    # recreate the image using reference mixing matrix
    Inorm = Io * np.exp(np.dot(-HERef, C))
    Inorm = np.reshape(Inorm.T, (M, N, 3), order='F')
    Inorm = np.minimum(np.round(Inorm), 255).astype('uint8')

    if return_he:
        # H = Io * np.exp(np.outer(-HERef[:, 0], C[0, :]))
        # H = np.reshape(H.T, (M, N, 3), order='F')
        # H = np.minimum(np.round(H), 255).astype('uint8')
        H = Io * np.exp(np.outer(-HERef[1, 0], C[0, :]))
        H = np.reshape(H.T, (M, N), order='F')
        H = np.minimum(np.round(H), 255).astype('uint8')

        # E = Io * np.exp(np.outer(-HERef[:, 1], C[1, :]))
        # E = np.reshape(E.T, (M, N, 3), order='F')
        # E = np.minimum(np.round(E), 255).astype('uint8')
        E = Io * np.exp(np.outer(-HERef[1, 1], C[1, :]))
        E = np.reshape(E.T, (M, N), order='F')
        E = np.minimum(np.round(E), 255).astype('uint8')
    else:
        H = []
        E = []

    return Inorm, H, E


def he_to_rgb(H, E, Io=240.0, HERef=np.array([(0.5626, 0.2159),
                                              (0.7201, 0.8012),
                                              (0.4062, 0.5581)])):
    H = H.astype("float32")
    E = E.astype("float32")
    M, N = H.shape
    h = np.reshape(H, M*N)
    e = np.reshape(H, M*N)
    C = np.zeros((2, M * N), dtype='float32')
    C[0, :] = np.log(h / Io) / (-HERef[1, 0])
    C[1, :] = np.log(e / Io) / (-HERef[1, 1])
    Inorm = Io * np.exp(np.dot(-HERef, C))
    Inorm = np.reshape(Inorm.T, (M, N, 3))
    Inorm = np.minimum(np.round(Inorm), 255).astype('uint8')

    return Inorm
