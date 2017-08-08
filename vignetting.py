import cv2
import numpy as np

img = cv2.cvtColor(cv2.imread('./images/dahlia.jpg'), cv2.COLOR_BGR2RGB)
rows, cols = img.shape[:2]

# generating vignette mask using Gaussian kernels
kernel_x = cv2.getGaussianKernel(cols, 150) # second parameter = standard deviation of the Gaussian
kernel_y = cv2.getGaussianKernel(rows, 150)
kernel = kernel_y * kernel_x.T # building 2D kernel whose size matches the size of the image.

mask = 255 * kernel / np.linalg.norm(kernel)
output = np.copy(img)

# applying the mask to each channel in the input image
for i in range(3):
    output[:,:,i] = output[:,:,i] * mask

cv2.imshow('Original', img)
cv2.imshow('Vignette', output)
cv2.waitKey(0)