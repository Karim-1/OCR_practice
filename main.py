import numpy as np
import pytesseract
import PIL.Image
import cv2 

print('test')

myconfig = r'--psm 4 --oem 3'

text = pytesseract.image_to_string(PIL.Image.open('data/bonnetje.jpeg'), config = myconfig)
print(text)
