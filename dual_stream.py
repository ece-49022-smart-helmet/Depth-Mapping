import cv2 as cv
import numpy as np

p1 = int(input('Port 1? '))
try:
    cam1 = cv.VideoCapture(p1)
    assert cam1.isOpened()
except Exception:
    print(f'{p1} is not a valid camera port!')
    raise SystemExit()

p2 = int(input('Port 2? '))
try:
    cam2 = cv.VideoCapture(p2)
    assert cam2.isOpened()
except Exception:
    print(f'{p2} is not a valid camera port!')
    raise SystemExit()

prefix1 = input('Camera 1 prefix? ')
prefix2 = input('Camera 2 prefix? ')

ix = 0


while True:
    while True:
        try:
            (_, img1), (_, img2) = cam1.read(), cam2.read()
            cv.imshow('frame', np.concatenate((img1, img2), axis=1))
            cv.waitKey(1)
        except KeyboardInterrupt:
            break
    cv.imwrite(f'{prefix1}{ix}.png', img1)
    cv.imwrite(f'{prefix2}{ix}.png', img2)
    print(f'Wrote out images {ix}!')
    ix += 1
    n = input()
    if n == 'quit':
        break

del cam1
del cam2

print('Bye!')
