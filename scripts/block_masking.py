"""
Figuring out how to isolate blocks and find their centroids from images of them on the floor
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from machinevisiontoolbox.base import *
from machinevisiontoolbox import *

# get image from cube
# path_to_image = './block_images/allColors.jpg'#???
path_to_image = './block_images/lookDownAtFloorwBlocks.jpg'#???
img = cv.imread(path_to_image) 

images = ImageCollection('block_images/*.jpg')

###### TODO: Calibrate images
K = np.array([[1524, 0, 350.5], [0, 987.1, 80.37], [ 0, 0, 1]])
    # matrix of camera's intrinsic parameters (K)
# extracting intrinsic parameters
u0 = K[0,2]
v0 = K[1,2]
fpixel_width = K[0,0]
fpixel_height = K[1,1]

distortion = [0.5675, -80.26, 0.8333, -0.001611, 874.7] 
    # lens distortion parameters
# extracting distortion parameters
k1, k2, p1, p2, k3 = distortion






# mask out floor from cube images
gaussianBlur = blur = cv.GaussianBlur(img,(5,5),0)
img_grayscale = cv.cvtColor(gaussianBlur, cv.COLOR_BGR2GRAY)
# Otsu's thresholding
ret, bw_thresh = cv.threshold(img_grayscale,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
bw_thresh = cv.bitwise_and(gaussianBlur, gaussianBlur, mask=bw_thresh)
closing = cv.morphologyEx(bw_thresh, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT,(5,5)))
openAfterClosing = cv.morphologyEx(closing, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT,(5,5)))

# get contours
finishedMask = cv.cvtColor(openAfterClosing, cv.COLOR_BGR2GRAY)
contours, hierarchy = cv.findContours(finishedMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(img, contours, -1, (255,0,255), 3)

# get bounding boxes and centroids

# filters contour list so that bounding boxes and centroids of only large enough contours are found
filteredContours = []
for contour in contours:
    if cv.contourArea(contour) > 200:
        filteredContours.append(contour)

boundingBoxes = []
centroids = []

# Iterate through contours
for contour in filteredContours:
    # Get bounding rectangle
    x, y, w, h = cv.boundingRect(contour)
    # Calculate centroid
    center_x = x + w // 2
    center_y = y + h // 2

    boundingBoxes.append([x, y, w, h])
    centroids.append([center_x, center_y])

    # Draw bounding box and centroid
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle
    cv.circle(img, (center_x, center_y), 5, (0, 255, 0), -1)  # Green circle for centroid

# Display the image with bounding boxes and centroids
print(f"bounding boxes: {boundingBoxes}")
print(f"centroids: {centroids}")
cv.imshow('boundingBoxes', img)
cv.waitKey(0)

# find position and orientation of blocks