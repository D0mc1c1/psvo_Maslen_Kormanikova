'''
from ximea import xiapi
import cv2
### runn this command first echo 0|sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb  ###

# create instance for first connected camera
cam = xiapi.Camera()

# start communication
# to open specific device, use:
# cam.open_device_by_SN('41305651')
# (open by serial number)
print('Opening first camera...')
cam.open_device()

# settings
cam.set_exposure(50000)
cam.set_param("imgdataformat","XI_RGB32")
cam.set_param("auto_wb",1)

print('Exposure was set to %i us' %cam.get_exposure())

# create instance of Image to store image data and metadata
img = xiapi.Image()

# start data acquisitionq
print('Starting data acquisition...')
cam.start_acquisition()


while cv2.waitKey() != ord('q'):
    cam.get_image(img)
    image = img.get_image_data_numpy()
    image = cv2.resize(image,(240,240))

    cv2.imshow("test", image)
    cv2.waitKey()   

# for i in range(10):
#     #get data and pass them from camera to img
#     cam.get_image(img)
#     image = img.get_image_data_numpy()
#     cv2.imshow("test", image)
#     cv2.waitKey()
#     #get raw data from camera
#     #for Python2.x function returns string
#     #for Python3.x function returns bytes
#     data_raw = img.get_image_data_raw()
#
#     #transform data to list
#     data = list(data_raw)
#
#     #print image data and metadata
#     print('Image number: ' + str(i))
#     print('Image width (pixels):  ' + str(img.width))
#     print('Image height (pixels): ' + str(img.height))
#     print('First 10 pixels: ' + str(data[:10]))
#     print('\n')

# stop data acquisition
print('Stopping acquisition...')
cam.stop_acquisition()

# stop communication
cam.close_device()

print('Done.')
'''


import cv2 as cv
import numpy as np
import sys

img1 = cv.imread("images/img1.jpg")
img2 = cv.imread("images/img2.jpg")
img3 = cv.imread("images/img3.jpg")
img4 = cv.imread("images/img4.jpg")

if img1 is None or img2 is None or img3 is None or img4 is None:
    sys.exit("Could not read one of the images.")

h, w, c = img1.shape

mozaika = np.zeros((2*h, 2*w, c), dtype=np.uint8)

mozaika[0:h, 0:w] = img1
mozaika[0:h, w:2*w] = img2
mozaika[h:2*h, 0:w] = img3
mozaika[h:2*h, w:2*w] = img4

cv.imshow("Mozaika", mozaika)
cv.imwrite("images/mozaika.jpg", mozaika)
k = cv.waitKey(0)
if k == 27:   # ESC
    cv.destroyAllWindows()
# print(img3.shape)

# uloha c.3
cast_1 = mozaika[0:h, 0:w]

kernel = np.array([[ 0,-1, 0],
                   [-1, 5,-1],
                   [ 0,-1, 0]], np.float32)

cast_1_filter = cv.filter2D(cast_1, -1, kernel, borderType=cv.BORDER_DEFAULT)

mozaika[0:h, 0:w] = cast_1_filter

'''
cv.imshow("Mozaika sharp", mozaika)
cv.imwrite("images/mozaika_filter.jpg", mozaika)
k = cv.waitKey(0)
if k == 27:   # ESC
    cv.destroyAllWindows()
'''

# uloha c.4
cast_2 = mozaika[0:h, w:2*w]

cast_2_rotated = np.zeros((w, h, c), dtype=np.uint8)

for i in range(h):
    for j in range(w):
        cast_2_rotated[j, h-1-i] = cast_2[i, j]

mozaika[0:h, w:2*w] = cast_2_rotated

'''
cv.imshow("Mozaika rotate", mozaika)
cv.imwrite("images/mozaika_filter.jpg", mozaika)
k = cv.waitKey(0)
if k == 27:   # ESC
    cv.destroyAllWindows()
'''

# uloha c.5
cast3 = mozaika[h:2*h, 0:w]

# BGR, nie RGB
cast3[:, :, 0] = 0   # Blue = 0
cast3[:, :, 1] = 0   # Green = 0

cv.imshow("Mozaika filtered", mozaika)
cv.imwrite("images/mozaika_filter.jpg", mozaika)
k = cv.waitKey(0)
if k == 27:   # ESC
    cv.destroyAllWindows()

# uloha c.6
print("Dtype:", mozaika.dtype)
print("Shape:", mozaika.shape)
print("Size:", mozaika.size)