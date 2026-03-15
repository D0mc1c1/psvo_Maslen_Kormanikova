from ximea import xiapi
import cv2 as cv
import numpy as np

import pickle

with open("calibration.pkl", "rb") as f:
    data = pickle.load(f)

mtx = data["camera_matrix"]
dist = data["dist_coeff"]

cam = xiapi.Camera()
cam.open_device()

# settings
cam.set_exposure(50000)
cam.set_param("imgdataformat", "XI_RGB32")
cam.set_param("auto_wb", 1)

img = xiapi.Image()
cam.start_acquisition()

while True:

    cam.get_image(img)
    frame = img.get_image_data_numpy()
    frame = cv.resize(frame,(616, 514))

    undistorted = cv.undistort(frame, mtx, dist, None, mtx)

    hsv = cv.cvtColor(undistorted, cv.COLOR_BGR2HSV)

    lower_red1 = np.array([0,120,70])
    upper_red1 = np.array([10,255,255])

    lower_red2 = np.array([170,120,70])
    upper_red2 = np.array([180,255,255])

    mask1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv.inRange(hsv, lower_red2, upper_red2)

    mask = mask1 + mask2

    undistorted[mask > 0] = (0,255,0)

    cv.imshow("Filtered image", undistorted)
    cv.imshow("Mask", mask)

    if cv.waitKey(1) == 27:
        break

cam.stop_acquisition()
cam.close_device()
cv.destroyAllWindows()