'''
main.py
Optical Character Recognition (OCR) practice
Karim Semin

$ tesseract --help-psm
Page segmentation modes:
  0    Orientation and script detection (OSD) only.
  1    Automatic page segmentation with OSD.
  2    Automatic page segmentation, but no OSD, or OCR. (not implemented)
  3    Fully automatic page segmentation, but no OSD. (Default)
  4    Assume a single column of text of variable sizes.
  5    Assume a single uniform block of vertically aligned text.
  6    Assume a single uniform block of text.
  7    Treat the image as a single text line.
  8    Treat the image as a single word.
  9    Treat the image as a single word in a circle.
 10    Treat the image as a single character.
 11    Sparse text. Find as much text as possible in no particular order.
 12    Sparse text with OSD.
 13    Raw line. Treat the image as a single text line,
       bypassing hacks that are Tesseract-specific.


OCR Engine modes: (see https://github.com/tesseract-ocr/tesseract/wiki#linux)
  0    Legacy engine only.
  1    Neural nets LSTM engine only.
  2    Legacy + LSTM engines.
  3    Default, based on what is available.
'''

import cv2 
import numpy as np
import PIL.Image
import pytesseract

from pytesseract import Output

# psm 12: Sparse text with OSD.
my_config = r'--psm 12 --oem 3'

# read image
img = cv2.imread('data/bonnetje.tiff')
# resize image because doesn't fit in window
img = cv2.resize(img, (800, 1422), interpolation=cv2.INTER_AREA)

# convert from RGB color-space to YCrCb
ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# equalize the histogram of the Y channel
ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

# convert back to RGB color-space from YCrCb
img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR) # <--- equalized contrast


# retrieve data from image
data = pytesseract.image_to_data(img, config = my_config, lang='nld', output_type=Output.DICT)

n_boxes = len(data['text']) # amount of boxes
threshold = 0 # confidence percentage threshold
for i in range(n_boxes):
    # only add rectangles and text if > confidence threshold
    if float(data['conf'][i]) > threshold:
        # get coordinates of rectangle from data
        (x, y, width, height) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])

        # add rectangle to image
        img = cv2.rectangle(img, 
            (x,y),
            (x+width, y+height), 
            color=(0,0,255), # BGR 
            thickness = 1) 
        # add text to image
        img = cv2.putText(img, 
            text = data['text'][i], 
            org = (x,y+height+20), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale = 0.6, 
            color = (0,0,255), 
            thickness=1,
            lineType = cv2.LINE_AA)

text = pytesseract.image_to_string(PIL.Image.open('data/bonnetje.jpeg'), lang='nld', config = my_config) 
print(text)

# show image with text and boxes
winname = "uitgelezen_bonnetje"
cv2.namedWindow(winname)        # Create a named window
cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
cv2.imshow(winname, img)
cv2.waitKey(0)


