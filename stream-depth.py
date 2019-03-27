"""
Gaussian blur original images instead of CLAHE or with CLAHE
play with params
setMin setSpeckleRange
"nearest neighbor" (four RLDU layers then take nonzero mean, 10+ times)
"""

import cv2
import signal
from sys import argv
import numpy as np
from time import time
from scipy import ndimage as nd
import _thread as thread

EXPOSURE = 0.08
BLUR = 11
M_FILTER = 11



print('Connecting to cameras...')
lcam = cv2.VideoCapture(0)
rcam = cv2.VideoCapture(1)

lcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
rcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

def cleanup(a, b):
    global lcam
    global rcam
    del lcam
    del rcam
    signal.alarm(0)
    cv2.destroyAllWindows()
    print('Bye...')
    raise SystemExit()

def singleDepthFrame(left, right, K1, K2, D1, D2, stereo):
    (_, limg), (_, rimg) = left.read(), right.read()
    lgray = cv2.cvtColor(limg, cv2.COLOR_BGR2GRAY)
    rgray = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)
    
    lgray = cv2.Canny(
        cv2.GaussianBlur(cv2.undistort(lgray, K1, D1), (BLUR, BLUR), 0), 50, 150)
    rgray = cv2.Canny(
        cv2.GaussianBlur(cv2.undistort(rgray, K2, D2), (BLUR, BLUR), 0), 50, 150)

    

    depth = stereo.compute(lgray, rgray)
    depth = cv2.GaussianBlur(depth, (M_FILTER, M_FILTER), 0)
    #filtered = nd.filters.median_filter(depth, (M_FILTER, M_FILTER))


    #lgray = cv2.Canny(cv2.undistort(lgray, K1, D1), 20, 200)
    #edges = cv2.Canny(cv2.undistort(rgray, K2, D2), 20, 200)
    
    #print('Filling image...')
    #ind = nd.distance_transform_edt(depth==-16, return_distances=False, return_indices=True)
    #filled = depth[tuple(ind)]
    
    return floatScale(depth), floatScale(lgray), floatScale(rgray)

def pollDepthFrame(a, b): # A and B parameters are just for signal call
    signal.alarm(2)
    print(signal.getsignal(signal.SIGALRM))
    depth, left, right = singleDepthFrame(lcam, rcam, K1, K2, D1, D2, stereo)
    cv2.imshow('', np.concatenate((depth, left, right), axis=1))
    cv2.waitKey(1)
    #print(depth.sum())

def clearBuffer():
    global lcam
    global rcam
    
    while True:
        pass
        #lcam.read()
        #rcam.read()

thread.start_new_thread(clearBuffer, ())

signal.signal(signal.SIGALRM, pollDepthFrame)
signal.signal(signal.SIGINT, cleanup)

print('Setting left camera exposure to {}...'.format(EXPOSURE))
lcam.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)

print('Setting right camera exposure to {}...'.format(EXPOSURE))
rcam.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)

print('Loading fisheye distortion parameters')
K1 = np.loadtxt('_left/K.txt')
D1 = np.loadtxt('_left/D.txt')
K2 = np.loadtxt('_right/K.txt')
D2 = np.loadtxt('_right/D.txt')

floatScale = lambda w: (w - w.min()) / (w.max() - w.min())
stereo = cv2.StereoBM_create(numDisparities=int(argv[1]), blockSize=int(argv[2]))

signal.alarm(1)

while True:
    #pollDepthFrame(0, 0)
    pass
