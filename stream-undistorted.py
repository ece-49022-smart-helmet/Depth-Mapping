import cv2
import numpy as np
import os

port1 = int(input('Left port? '))

try:
    lcam = cv2.VideoCapture(port1)
    assert lcam.isOpened()
except Exception as e:
    print('Failed opening a camera on port {}'.format(port1))
    print(e)
    raise SystemExit()

port2 = int(input('Right port? '))

try:
    rcam = cv2.VideoCapture(port2)
    assert rcam.isOpened()
except Exception as e:
    print('Failed opening a camera on port {}'.format(port2))
    print(e)
    raise SystemExit()


leftPath = input('Path to left camera distortion matrices? ')
try:
    K1 = np.loadtxt(os.path.join(leftPath, 'K.txt'))
    D1 = np.loadtxt(os.path.join(leftPath, 'D.txt'))
except Exception as e:
    print('Could not load matrices from {}'.format(leftPath))
    print(e)
    raise SystemExit()

rightPath = input('Path to right camera distortion matrices? ')
try:
    K2 = np.loadtxt(os.path.join(rightPath, 'K.txt'))
    D2 = np.loadtxt(os.path.join(rightPath, 'D.txt'))
except:
    print('Could not load matrices from {}'.format(rightPath))

while True:
    try:
        (lr, lim), (rr, rim) = lcam.read(), rcam.read()
        if not (lr and rr): continue
        cv2.imshow('Distorted', np.concatenate((lim, rim), axis=1))
        lim = cv2.undistort(lim, K1, D1)
        rim = cv2.undistort(rim, K2, D2)
        cv2.imshow('Corrected', np.concatenate((lim, rim), axis=1))
        cv2.waitKey(1)
    except KeyboardInterrupt:
        break
