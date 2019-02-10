import cv2
import numpy as np
import matplotlib.pyplot as plt

im = cv2.imread("../../panda.jpeg")


im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
print(im.shape)


plt.figure()
plt.imshow(im, cmap="gray")
plt.show()
