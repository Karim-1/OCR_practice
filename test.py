import cv2 
import numpy as np
import PIL.Image
import pytesseract

from pytesseract import Output



rgb_img = cv2.imread('data/bonnetje.tiff')

# convert from RGB color-space to YCrCb
ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YCrCb)

# equalize the histogram of the Y channel
ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

# convert back to RGB color-space from YCrCb
equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

cv2.imshow('equalized_img', equalized_img)
cv2.waitKey(0)

