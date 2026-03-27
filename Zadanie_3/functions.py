import numpy as np

class canny_algorithm:

    def my_conv2(self, img, kernel):
        img = img.astype(float)

        kH, kW = kernel.shape
        iH, iW = img.shape

        padH = kH // 2
        padW = kW // 2

        padded = np.zeros((iH + 2*padH, iW + 2*padW))
        padded[padH:padH+iH, padW:padW+iW] = img

        out = np.zeros((iH, iW))

        kernel = np.flipud(np.fliplr(kernel))

        for i in range(iH):
            for j in range(iW):

                region = padded[i:i+kH, j:j+kW]

                out[i, j] = np.sum(region * kernel)

        return out