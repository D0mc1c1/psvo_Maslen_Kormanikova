import numpy as np
import cv2 as cv
import glob
import pickle

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# images = glob.glob('images/img50.jpg')  pre konkretny obrazok
images = glob.glob('images/img*.jpg')   # pre viac obrázkov na kalibráciu

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,5), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,5), corners2, ret)
        cv.imshow('Corners', img)
        cv.waitKey(500)

# cv.destroyAllWindows()

# kalibracia
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

img = cv.imread("images/img6.jpg")
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)

cv.imshow("Original", img)
cv.imshow("Undistort", dst)
cv.waitKey(0)
cv.destroyAllWindows()

print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

fx = mtx[0,0]
print("fx:", fx)

fy = mtx[1,1]
print("fy:", fy)

cx = mtx[0,2]
print("cx:", cx)

cy = mtx[1,2]
print("cy:", cy)

np.savez("calibration_data.npz", mtx=mtx, dist=dist)

calibration_data = {
    "camera_matrix": mtx,
    "dist_coeff": dist
}

with open("calibration.pkl", "wb") as f:
    pickle.dump(calibration_data, f)

k = cv.waitKey(0)
if k == 27:   # ESC
    cv.destroyAllWindows()