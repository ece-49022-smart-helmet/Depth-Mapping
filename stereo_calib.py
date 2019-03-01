import numpy as np
import cv2
from glob import glob

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#limgs = glob('LEFT/*_UD.jpg')
#rimgs = glob('RIGHT/*_UD.jpg')

limgs = glob('LEFT/*.png')
rimgs = glob('RIGHT/*.png')

limgs.sort()
rimgs.sort()

objp = np.zeros((8*6,3), np.float32)
objp[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1,2)

objpts = []
imlpts = []
imrpts = []

for l, r in zip(limgs, rimgs):
    limg = cv2.imread(l)
    rimg = cv2.imread(r)

    lgrey = cv2.cvtColor(limg, cv2.COLOR_BGR2GRAY)
    rgrey = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)

    lret, lcorn = cv2.findChessboardCorners(lgrey, (8,6), None)
    rret, rcorn = cv2.findChessboardCorners(rgrey, (8,6), None)

    if not (lret and rret): continue
    objpts.append(objp)

    lcorn2 = cv2.cornerSubPix(lgrey, lcorn, (11,11), (-1,-1), criteria)
    rcorn2 = cv2.cornerSubPix(rgrey, rcorn, (11,11), (-1,-1), criteria)
    imlpts.append(lcorn2)
    imrpts.append(rcorn2)

#ret, K1, D1, _, _ = cv2.calibrateCamera(objpts, imlpts, lgrey.shape[::-1], None, None)
#ret, K2, D2, _, _ = cv2.calibrateCamera(objpts, imrpts, lgrey.shape[::-1], None, None)

#print(K1)
#print(K2)

flags = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_SAME_FOCAL_LENGTH | cv2.CALIB_RATIONAL_MODEL
ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objpts, imlpts, imrpts,
    None, None, None, None, lgrey.shape[::-1],
    criteria=criteria,
    flags=flags)
print(K1)
print(K2)

print()

print(D1.shape)
print(D2.shape)

print()

print(R)
print(T)

im1 = cv2.imread('LEFT/left6.png')
im2 = cv2.imread('RIGHT/right6.png')

#cv2.namedWindow('tuner')
#cv2.createTrackbar('alpha', 'tuner', 0, 100, lambda *x:0)
while False:
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, (640, 480), R, T,
        alpha=-1)#cv2.getTrackbarPos('alpha', 'tuner') / 100)
    lmap1, lmap2 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (640, 480), cv2.CV_16SC2)
    rmap1, rmap2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (640, 480), cv2.CV_16SC2)

    rim1 = cv2.remap(im1, lmap1, lmap2, cv2.INTER_LANCZOS4)
    rim2 = cv2.remap(im2, rmap1, rmap2, cv2.INTER_LANCZOS4)

    cv2.imshow('tuner', np.concatenate((rim1, rim2), axis=1))
    cv2.waitKey(0)

cv2.destroyAllWindows()

#print(roi1, roi2)





#K1 = np.loadtxt('LEFT/N.txt')
#K2 = np.loadtxt('RIGHT/N.txt')

ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objpts, imlpts, imrpts, None, None, None, None, lgrey.shape[::-1])

print(R)
print(T)

R1 = np.zeros((3,3))
R2 = np.zeros((3,3))

P1 = np.zeros((3,4))
P2 = np.zeros((3,4))

print(D1, D2)
print(K1, K2)
print(lgrey.shape[::-1])

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                                            np.eye(3), np.zeros((1,5)),
                                            np.eye(3), np.zeros((1,5)),
                                            lgrey.shape[::-1], R, T, -1, lgrey.shape[::-1] * 2) #, R1, R2, P1, P2)


print(roi1)
print(roi2)

print(R1)
print(R2)

print(P1)
print(P2)



limg = cv2.imread('LEFT/left11.png')
rimg = cv2.imread('RIGHT/right11.png')

print(limg.dtype)

pad = np.zeros((limg.shape), dtype=np.uint8)

#limg = np.concatenate((pad, limg, pad))
#rimg = np.concatenate((pad, rimg, pad))

print(limg.shape[:-1][::-1])

_p1, _ = cv2.getOptimalNewCameraMatrix(P1[:, :3], D1, limg.shape[:-1][::-1], 1, limg.shape[:-1][::-1])
_p2, _ = cv2.getOptimalNewCameraMatrix(P2[:, :3], D2, rimg.shape[:-1][::-1], 1, rimg.shape[:-1][::-1])


(lmap1, lmap2) = cv2.initUndistortRectifyMap(K1, D1, R1, P1, tuple(lgrey.shape[::-1]), cv2.CV_16SC2)
(rmap1, rmap2) = cv2.initUndistortRectifyMap(K2, D2, R2, P2, tuple(lgrey.shape[::-1]), cv2.CV_16SC2)

lster = np.zeros(limg.size)
rster = np.zeros(rimg.size)

pad = np.zeros(limg.shape, dtype=np.uint8)
#limg = np.concatenate((pad, limg, pad))
#rimg = np.concatenate((pad, rimg, pad))
lster = cv2.remap(limg, lmap1, lmap2, cv2.INTER_LANCZOS4)
rster = cv2.remap(rimg, rmap1, rmap2, cv2.INTER_LANCZOS4)

cv2.imshow('oldfucks', np.concatenate((limg, rimg), axis=1))
cv2.waitKey(0)
cv2.imshow('rectified', np.concatenate((lster, rster), axis=1))
cv2.waitKey(0)
cv2.destroyAllWindows()
