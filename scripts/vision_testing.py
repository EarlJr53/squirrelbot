"""
Initial code to interface with camera taken from Microcontrollers Explained
on YouTube: https://youtu.be/OEGR19aE0dg?si=H1v4T3LscPZ-eNlf
"""

import cv2
import time

num = 1
cap = cv2.VideoCapture(0)
    # index denotes camera, which we only have one of

while True:
    ret, img = cap.read()
    cv2.imshow('Frame', img)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        cv2.imwrite('../media/images/'+str(num)+'.jpg', img)
        print('Capture '+str(num)+' Successful!')
        num = num + 1
    if num == 4:
        break
cap.release()
cv2.destroyAllWindows()
