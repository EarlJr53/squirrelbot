"""
Figuring out how to isolate blocks and find their centroids from images of them on the floor
"""
import cv2 as cv
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import math
from machinevisiontoolbox.base import *
from machinevisiontoolbox import *

# get image from cube
# path_to_image = './block_images/allColors.jpg'#???
# path_to_image = './block_images/lookDownAtFloorwBlocks.jpg'#???
# img = cv.imread(path_to_image) 

images = ImageCollection('./new_angle/*.jpg') 
    # must be in this format for calibration to work, can't use meshgrid()
    # method with 'numpy.ndarray' object which cv.imread() gives
img = images[5]



#######################
# undistorting image: #
#######################
"""
img coming into this part must be an image object of the machinevisiontoolbox
"""
K = np.array([[460.2, 0, 350.6], [0, 452.4, 235.7], [0, 0, 1]])
    # matrix of camera's intrinsic parameters (K)
# extracting intrinsic parameters
u0 = K[0,2]
v0 = K[1,2]
fpixel_width = K[0,0]
fpixel_height = K[1,1]

distortion = [-0.4033, 0.2033, 0.00473, 0.001013, -0.05674]  
    # lens distortion parameters
# extracting distortion parameters
k1, k2, p1, p2, k3 = distortion

# Convert from pixel coordinates (u, v) to image plane coordinates (x, y)
U, V = img.meshgrid()
x = (U - u0) / fpixel_width
y = (V - v0) / fpixel_height

# Calculate the radial distance of pixels from the principal point
r = np.sqrt(x**2 + y**2)

# Compute the image coordinate errors due to both radial and tangential distortion
delta_x = x * (k1*r**2 + k2*r**4 + k3*r**6) + 2*p1*x*y + p2*(r**2 + 2*x**2)
delta_y = y * (k1*r**2 + k2*r**4 + k3*r**6) + p1*(r**2 + 2*y**2) + p2*x*y

# Distorted retinal coordinates
xd = x + delta_x 
yd = y + delta_y

# Convert back from image coordinates to pixel coordinates in the distorted image
Ud = xd * fpixel_width + u0
Vd = yd * fpixel_height + v0

# Apply the warp to a distorted image and observe the undistorted image
img = img.warp(Ud, Vd)


#####################
# gamma correction: #
#####################
"""
img coming into this part is assumed to be an image object of the machinevisiontoolbox
could comment out first line below this if already a numpy array 

gamma correction code taken from https://pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    applies a Power Law Transform where O = I ^ (1 / G)
    - O: output image, mapped to a 0-225 scale at the end
    - I: input image, converted to a 0-1.0 scale at the beginning
    - G: gamma value, where G < 1 makes image darker and G > 1 makes image brighter

not really necessary with a large opening kernel in morphology operations below, 
but should help make masking slightly more robust
"""
# convert image to numpy array
img = img.image

gamma = 0.5 # will darken image

invGamma = 1.0 / gamma

# create lookup table to map
table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
# apply gamma correction using the lookup table
img = cv.LUT(img, table)


####################################
# masking and contour shenanigans: #
####################################
"""

"""
# mask out floor from cube images
gaussianBlur = blur = cv.GaussianBlur(img,(5,5),0)
img_grayscale = cv.cvtColor(gaussianBlur, cv.COLOR_BGR2GRAY)
# Otsu's thresholding
ret, bw_thresh = cv.threshold(img_grayscale,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
bw_thresh = cv.bitwise_and(gaussianBlur, gaussianBlur, mask=bw_thresh)
closing = cv.morphologyEx(bw_thresh, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT,(5,5)))
openAfterClosing = cv.morphologyEx(closing, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT,((20,20))))
    # dramatically increased the size of the kernel from a (5,5) rectangle to a (20,20) rectangle

# get contours
finishedMask = cv.cvtColor(openAfterClosing, cv.COLOR_BGR2GRAY)
contours, hierarchy = cv.findContours(finishedMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(img, contours, -1, (255,0,255), 3)

# get bounding boxes and centroids

# filters contour list so that bounding boxes and centroids of only large enough contours are found
filteredContours = []
for contour in contours:
    if cv.contourArea(contour) > 10000:
        # 10000 area should count only blocks which are just about fully visible
        # (and therefore have reliable enough centroids)
        filteredContours.append(contour)

print(f"filtered contours: {filteredContours}")

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


###############################
# pos of blocks in cam frame: #
###############################
inv_K = np.linalg.inv(K)
# print(K)
# print(inv_K)

# appending the normalization factor
w = 1 # normalization factor
chosen_block = 0 # index for centroids list
centroids[chosen_block].append(w)
# print(centroids[0])

cam_coords = np.matmul(inv_K, centroids[0])
print(f"coordinates in camera frame: {cam_coords}")