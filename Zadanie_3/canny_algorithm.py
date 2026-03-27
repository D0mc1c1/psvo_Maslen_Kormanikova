import numpy as np
import cv2
import matplotlib.pyplot as plt

from functions import canny_algorithm
functions = canny_algorithm()

img = cv2.imread("cameraman.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img,(616, 514))

G = (1/16) * np.array([
    [1,2,1],
    [2,4,2],
    [1,2,1]
])

img = img.astype(float)
blur = functions.my_conv2(img, G)

Sx = np.array([
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]
])

Sy = np.array([
    [-1,-2,-1],
    [0,0,0],
    [1,2,1]
])

Gx = functions.my_conv2(blur, Sx)
Gy = functions.my_conv2(blur, Sy)

mag = np.sqrt(Gx**2 + Gy**2)
theta = np.arctan2(Gy, Gx)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Canny algorithm")
plt.imshow(mag, cmap="gray")
plt.axis("off")

plt.show()