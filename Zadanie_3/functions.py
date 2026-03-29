import numpy as np

class canny_algorithm:

    def my_conv2(self, img, kernel):
        img = img.astype(float)

        kH, kW = kernel.shape
        iH, iW = img.shape

        padded_H = kH // 2
        padded_W = kW // 2

        padded = np.zeros((iH + 2*padded_H, iW + 2*padded_W))
        padded[padded_H:padded_H+iH, padded_W:padded_W+iW] = img

        output = np.zeros((iH, iW))

        kernel = np.flipud(np.fliplr(kernel))

        for i in range(iH):
            for j in range(iW):

                region = padded[i:i+kH, j:j+kW]
                output[i, j] = np.sum(region * kernel)

        return output
    
    def non_max_suppression(self, MAG, theta):

        H, W = MAG.shape
        output = np.zeros((H, W))

        angle = theta * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1, H-1):
            for j in range(1, W-1):

                A = 255
                B = 255

                # 0°
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    A = MAG[i, j+1]
                    B = MAG[i, j-1]

                # 45°
                elif (22.5 <= angle[i,j] < 67.5):
                    A = MAG[i+1, j-1]
                    B = MAG[i-1, j+1]

                # 90°
                elif (67.5 <= angle[i,j] < 112.5):
                    A = MAG[i+1, j]
                    B = MAG[i-1, j]

                # 135°
                elif (112.5 <= angle[i,j] < 157.5):
                    A = MAG[i-1, j-1]
                    B = MAG[i+1, j+1]

                if (MAG[i,j] >= A) and (MAG[i,j] >= B):
                    output[i,j] = MAG[i,j]
                else:
                    output[i,j] = 0

        return output
    
    def double_threshold(self, img, low, high):

        H, W = img.shape
        output = np.zeros((H, W))

        strong = 255
        weak = 75

        for i in range(H):
            for j in range(W):

                if img[i,j] >= high:
                    output[i,j] = strong

                elif img[i,j] >= low:
                    output[i,j] = weak

                else:
                    output[i,j] = 0

        return output
    
    def hysteresis(self, img):

        H, W = img.shape

        strong = 255
        weak = 75

        output = img.copy()

        for i in range(1, H-1):
            for j in range(1, W-1):

                if output[i, j] == weak:

                    # kontrola 8 susedov
                    if (
                        (output[i+1, j-1] == strong) or
                        (output[i+1, j] == strong) or
                        (output[i+1, j+1] == strong) or
                        (output[i, j-1] == strong) or
                        (output[i, j+1] == strong) or
                        (output[i-1, j-1] == strong) or
                        (output[i-1, j] == strong) or
                        (output[i-1, j+1] == strong)
                    ):

                        output[i, j] = strong

                    else:
                        output[i, j] = 0

        return output
