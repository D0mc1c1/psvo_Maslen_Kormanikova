from ximea import xiapi
import cv2

import matplotlib.pyplot as plt

'''
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
    image = cv2.resize(frame,(616, 514))

    cv2.imshow("Camera", image)

    key = cv2.waitKey(1)

    if key == 32:  
        filename = "image.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")

        break

    if key == 27: # ESC
        break

print('Stopping acquisition...')
cam.stop_acquisition()
cam.close_device()
cv2.destroyAllWindows()
'''

img = cv2.imread("cameraman.jpg", cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img, 50, 100)

plt.title("Original Image")
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()

plt.title("OpenCV Canny")
plt.imshow(edges, cmap="gray")
plt.axis("off")
plt.show()