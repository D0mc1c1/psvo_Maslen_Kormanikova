from ximea import xiapi
import cv2
import numpy as np

from functions import canny_algorithm
functions = canny_algorithm()

cam = xiapi.Camera()
cam.open_device()

# settings
cam.set_exposure(50000)
cam.set_param("imgdataformat", "XI_RGB32")
cam.set_param("auto_wb", 1)

img = xiapi.Image()

# start data acquisitionq
print('Starting data acquisition...')
cam.start_acquisition()

while True:
    cam.get_image(img)
    frame = img.get_image_data_numpy()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(616, 514))

    Gauss = (1/16) * np.array([
    [1,2,1],
    [2,4,2],
    [1,2,1]
    ])

    blur = functions.my_conv2(image, Gauss)

    # vertikalny sobelov filter
    Sobel_V = np.array([
        [-1,0,1],
        [-2,0,2],
        [-1,0,1]
    ])

    # horizontalny sobelov filter
    Sobel_H = np.array([
        [-1,-2,-1],
        [0,0,0],
        [1,2,1]
    ])

    G_H = functions.my_conv2(blur, Sobel_V)
    G_V = functions.my_conv2(blur, Sobel_H)

    MAG = np.sqrt(G_H**2 + G_V**2)
    theta = np.arctan2(G_V, G_H)

    NM_Supp = functions.non_max_suppression(MAG, theta)
    dt = functions.double_threshold(NM_Supp, 50, 100)
    edges = functions.hysteresis(dt)

    cv2.imshow("Original", image)
    cv2.imshow("Canny", edges)

    k = cv2.waitKey(1)
    if k == 27:   # ESC
        cam.stop_acquisition()
        cam.close_device()
        cv2.destroyAllWindows()