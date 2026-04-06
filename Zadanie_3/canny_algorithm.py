import numpy as np
import cv2
import matplotlib.pyplot as plt

from functions import canny_algorithm
functions = canny_algorithm()

img = cv2.imread("cameraman.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img,(616, 514))

Gauss = (1/16) * np.array([
    [1,2,1],
    [2,4,2],
    [1,2,1]
])

blur = functions.my_conv2(img, Gauss)

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


# originál
plt.title("Original")
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()

# blur
plt.title("Gaussian Blur")
plt.imshow(blur, cmap="gray")
plt.axis("off")
plt.show()

# gradient
plt.title("Gradient Magnitude")
plt.imshow(MAG, cmap="gray")
plt.axis("off")
plt.show()

# NMS
plt.title("NMS")
plt.imshow(NM_Supp, cmap="gray")
plt.axis("off")
plt.show()

# double threshold
plt.title("Double Threshold")
plt.imshow(dt, cmap="gray")
plt.axis("off")
plt.show()

# hysteresis — finálny výsledok
plt.title("Final Canny")
plt.imshow(edges, cmap="gray")
plt.axis("off")
plt.show()