from ximea import xiapi
import cv2
import os

# vytvorenie priecinka
if not os.path.exists("images"):
    os.makedirs("images")

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
cam.set_param("imgdataformat", "XI_RGB32")
cam.set_param("auto_wb", 1)

img = xiapi.Image()

# start data acquisitionq
print('Starting data acquisition...')
cam.start_acquisition()

count = 0

while True:
    cam.get_image(img)
    image = img.get_image_data_numpy()
    image = cv2.resize(image,(240,240))

    cv2.imshow("Camera", image)

    key = cv2.waitKey(1)

    if key == 32:
        filename = f"images/img{count+1}.jpg"
        cv2.imwrite(filename, image)
        print(f"Saved {filename}")
        count += 1

    if count == 4:
        break

    if key == 27: # ESC
        break

print('Stopping acquisition...')
cam.stop_acquisition()
cam.close_device()
cv2.destroyAllWindows()

print('Done saving images.')