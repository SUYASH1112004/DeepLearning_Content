#The Code displays how images are stored in pixel grids
import cv2
import numpy as np
import matplotlib.pyplot as plt

#1.Read Image
img = cv2.imread('photo.jpg')

#2 Convert to grayscale
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#3 Display Image + Pixel Grid
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(gray,cmap='gray')
plt.title("Gray Scale Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(gray,cmap='gray')
plt.colorbar(label="Pixel Value")
plt.title("Pixel Values")
plt.show()