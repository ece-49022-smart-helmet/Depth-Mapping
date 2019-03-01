import cv2 as cv
import numpy as np
from glob import glob


def clean(img, cl):
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    small = cv.resize(grey, (0,0), fx=0.5, fy=0.5)
    return cl.apply(small)

CB_SIZE = (8,6)

lpics = glob('LEFT/*.png')
rpics = glob('RIGHT/*.png')

print(lpics)
print(rpics)


criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

pointset = np.zeros((CB_SIZE[0]*CB_SIZE[1], 3), np.float32)
pointset[:, :2] = np.mgrid[0:CB_SIZE[0], 0:CB_SIZE[1]].T.reshape(-1, 2)

objpts = []
lmgpts = []
rmgpts = []

clahe = cv.createCLAHE()

for lf, rf in zip(lpics, rpics):
    limg = cv.imread(lf)
    rimg = cv.imread(rf)

    lgrey = clean(limg, clahe)
    rgrey = clean(rimg, clahe)

    lret, lcor = cv.findChessboardCorners(lgrey, CB_SIZE, None)
    rret, rcor = cv.findChessboardCorners(rgrey, CB_SIZE, None)
    print(lf, rf, lret, rret)
    if not (lret and rret): continue

    objpts.append(pointset)

    lcor2 = cv.cornerSubPix(lgrey, lcor, (11,11), (-1,-1), criteria)
    rcor2 = cv.cornerSubPix(rgrey, rcor, (11,11), (-1,-1), criteria)

    lmgpts.append(lcor2)
    rmgpts.append(rcor2)

    print(f'Calibrated with {lf}, {rf}')

ret, K1, D1, R1, T1 = cv.calibrateCamera(objpts, lmgpts, lgrey.shape[::-1], None, None)
ret, K2, D2, R2, T2 = cv.calibrateCamera(objpts, rmgpts, rgrey.shape[::-1], None, None)

#left = cv.imread('left0.png')
#right = cv.imread('right0.png')

#h,w =left.shape[:2]

#M1, _ = cv.getOptimalNewCameraMatrix(K1, D1, (w,h), 1, (w,h))
#M2, _ = cv.getOptimalNewCameraMatrix(K2, D2, (w,h), 1, (w,h))

#ldst = cv.undistort(left, K1, D1, None, M1)
#rdst = cv.undistort(right, K2, D2, None, M2)

#cv.imshow('undistorted', np.concatenate((ldst, rdst), axis=1))
#cv.waitKey(0)
#cv.destroyAllWindows()

N_OK = len(lmgpts)

objpts = np.array([pointset] * len(lmgpts), dtype=np.float64)
lmgpts = np.asarray(lmgpts, dtype=np.float64)
rmgpts = np.asarray(rmgpts, dtype=np.float64)

objpts = np.reshape(objpts, (N_OK, 1, CB_SIZE[0]*CB_SIZE[1], 3))
lmgpts = np.reshape(lmgpts, (N_OK, 1, CB_SIZE[0]*CB_SIZE[1], 2))
rmgpts = np.reshape(lmgpts, (N_OK, 1, CB_SIZE[0]*CB_SIZE[1], 2))

cv.fisheye.stereoCalibrate(objpts,lmgpts, rmgpts, None, None, None, None, lgrey.shape[::-1], flags=(cv.CALIB_FIX_ASPECT_RATIO + 
cv.CALIB_ZERO_TANGENT_DIST + cv.CALIB_SAME_FOCAL_LENGTH), criteria=(cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 100, 1e-5))


#raise SystemExit()
"""

K1 = np.zeros((3,3))
K2 = np.zeros((3,3))

D1 = np.zeros((4,1))
D2 = np.zeros((4,1))

R = np.zeros((1,1,3), dtype=np.float64)
T = np.zeros((1,1,3), dtype=np.float64)

ret, K1, D1, K2, D2, R, T = cv.fisheye.stereoCalibrate(objpts, lmgpts, rmgpts, K1, np.matrix([]), K2, np.matrix([]), 
lgrey.shape[::-1],flags=0)
#    K1, D1,
#    K2, D2, lgrey.shape[::-1], None, None,
#    cv.CALIB_FIX_INTRINSIC)


print('Left calibration: ')
print(K1, D1)
print('Right calibration: ')
print(K2, D2)

print('Camera rotation: ')
print(R)

print('Camera translation: ')
print(T)
"""
