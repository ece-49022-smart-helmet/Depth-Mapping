from glob import glob
import os
import cv2
import numpy as np

pics = glob(input('Calibration images pattern? '))

SZ = (8, 6)

# termination criteria
crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

pointset = np.zeros((SZ[0]*SZ[1], 3), dtype=np.float32)
pointset[:,:2] = np.mgrid[:SZ[0], :SZ[1]].T.reshape(-1, 2)


objpts = []
imgpts = []

for pic in pics:
    img = cv2.imread(pic)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corn = cv2.findChessboardCorners(gray, SZ, None)
    if not ret: continue
    
    print('Successfully calibrated with {}'.format(pic.split('/')[-1]))
    objpts.append(pointset)
    
    corn2 = cv2.cornerSubPix(gray, corn, (11, 11), (-1, -1), crit)
    imgpts.append(corn2)

print('Generating dunistortion matrices...')
ret, K, D, R, T = cv2.calibrateCamera(objpts, imgpts, gray.shape[::-1], None, None)
h, w = gray.shape
print('Optimizing undistortion matrices...')
K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))

path = input('Path to save undistortion matrices? ')
np.savetxt(os.path.join(path, 'K.txt'), K)
np.savetxt(os.path.join(path, 'D.txt'), D)
