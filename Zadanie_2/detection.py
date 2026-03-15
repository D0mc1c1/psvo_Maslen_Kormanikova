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
    frame = frame[:,:,:3]   # otestovať v skole kolko kanalov ma kamera realne
    frame = cv.resize(frame,(616, 514))
    # print(frame.shape) # pridat iba na otestovanie

    undistorted = cv.undistort(frame, mtx, dist, None, mtx)

    gray = cv.cvtColor(undistorted, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (9,9), 2)

    # detection circles
    circles = cv.HoughCircles(blur,cv.HOUGH_GRADIENT,1,20,
                            param1=50,param2=60,minRadius=0,maxRadius=0)

    if circles is not None:

        circles = np.uint16(np.around(circles))

        for i in circles[0,:]:
            # draw the outer circle
            cv.circle(undistorted,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv.circle(undistorted,(i[0],i[1]),2,(0,0,255),3)

            cx = i[0]
            cy = i[1]
            roi = undistorted[cy-5:cy+5, cx-5:cx+5]

            b,g,r = np.mean(roi,axis=(0,1))

            color = "Unknown"

            if r > 150 and g < 100 and b < 100:
                color = "Red"

            elif g > 150 and r < 100 and b < 100:
                color = "Green"

            elif b > 150 and r < 100 and g < 100:
                color = "Blue"

            elif r > 150 and g > 150:
                color = "Yellow"

            print("Detected: Circle with color:",color)

    # detection edges
    _, thresh = cv.threshold(gray,120,255,cv.THRESH_BINARY)
    edges = cv.Canny(thresh,50,150)

    contours,_ = cv.findContours(edges,
                                 cv.RETR_EXTERNAL,
                                 cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:

        area = cv.contourArea(cnt)

        if area < 500:
            continue

        peri = cv.arcLength(cnt,True)
        approx = cv.approxPolyDP(cnt,0.04*peri,True)

        vertices = len(approx)

        shape = "Unknown"

        if vertices == 3:
            shape = "Triangle"

        elif vertices == 4:

            x,y,w,h = cv.boundingRect(approx)
            ratio = w/float(h)

            if 0.95 <= ratio <= 1.05:
                shape = "Square"
            else:
                shape = "Rectangle"

        if shape != "Unknown":

            M = cv.moments(cnt)

            if M["m00"] == 0:
                continue

            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])

            cv.drawContours(undistorted,[approx],-1,(0,255,0),2)
            cv.circle(undistorted,(cx,cy),4,(0,0,255),-1)

            roi = undistorted[cy-5:cy+5, cx-5:cx+5]

            b,g,r = np.mean(roi,axis=(0,1))

            color = "Unknown"

            if r > 150 and g < 100 and b < 100:
                color = "Red"

            elif g > 150 and r < 100 and b < 100:
                color = "Green"

            elif b > 150 and r < 100 and g < 100:
                color = "Blue"

            elif r > 150 and g > 150:
                color = "Yellow"

            print("Detected:",shape,"with color:",color)

 
    cv.imshow("Shapes detection",undistorted)
    cv.imshow("Edges",edges)

    if cv.waitKey(1) == 27:
        break

cam.stop_acquisition()
cam.close_device()
cv.destroyAllWindows()