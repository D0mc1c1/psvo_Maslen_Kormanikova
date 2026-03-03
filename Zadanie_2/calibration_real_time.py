from ximea import xiapi
import cv2 as cv

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

    cv.imshow("Original", frame)
    cv.imshow("Undistorted", undistorted)

    k = cv.waitKey(1)
    if k == 27:   # ESC
        cam.stop_acquisition()
        cam.close_device()
        cv.destroyAllWindows()
