import numpy as np
import cv2
from glob import glob

CB_SZ = (8,6)

# termination criteria
crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CB_SZ[0]*CB_SZ[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CB_SZ[0], 0:CB_SZ[1]].T.reshape(-1, 2)

objpts = []
imgpts = []

pics = glob('LEFT/*.png')

for pic in pics:
    img = cv2.imread(pic)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CB_SZ, None)
    if not ret: continue

    objpts.append(objp)
    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), crit)
    imgpts.append(corners2)
    cv2.drawChessboardCorners(img, CB_SZ, corners2, ret)
    cv2.imshow('cb', img)
    cv2.waitKey(500)



cv2.destroyAllWindows()

# print(objpts, len(objpts))
# print(imgpts, len(imgpts))

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, gray.shape[::-1], None, None)

print(mtx, dist)

for pic in pics:
    test_image = cv2.imread(pic)
    h,w = test_image.shape[:2]
    newmtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    print(newmtx)
    dst = cv2.undistort(test_image, mtx, dist) #, None, newmtx)
    cv2.imwrite(pic[:-4] + '_UD.jpg', dst)
np.savetxt('LEFT/K.txt', mtx)
np.savetxt('LEFT/N.txt', newmtx)
