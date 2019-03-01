import cv2
import numpy as np

port1 = int(input('Left port? '))

try:
    lcam = cv2.VideoCapture(port1)
    assert lcam.isOpened()
except Exception as e:
    print('Failed opening a camera on port {}'.format(port1))
    print(e)

port2 = int(input('Right port? '))

try:
    rcam = cv2.VideoCapture(port2)
    assert rcam.isOpened()
except Exception as e:
    print('Failed opening a camera on port {}'.format(port2))
    print(e)


while True:
    try:
        (lr, lim), (rr, rim) = lcam.read(), rcam.read()
        if not (lr and rr): continue
        cv2.imshow('Frame', np.concatenate((lim, rim), axis=1))
        cv2.waitKey(1)
    except KeyboardInterrupt:
        break

del lcam
del rcam
print('Bye!')
