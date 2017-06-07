import numpy as np
import cv2
import copy
import os

img = cv2.imread("Nokia.jpg", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 240, 255,cv2.THRESH_BINARY_INV)


kernel = np.ones((25,25), np.uint8)
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

blur = cv2.GaussianBlur(morph, (15,15), 0)

corners = cv2.goodFeaturesToTrack(morph, 100, 0.01, 50)
print("Number of vertices = " + str(len(corners)))
corners = np.int0(corners)

for corner in corners:
    x, y = corner.ravel()   
    cv2.circle(img, (x,y), 3, 255, -1)

if(len(corners) == 3):
    print("Traingle")
elif(len(corners) == 4):
    print("Quadrilateral")


#:::::::::::Scraps::::::::::
#peri = cv2.arcLength(thresh, True)
#approx = cv2.approxPolyDP(thresh, 0.04 * peri, True)
#print(approx)
#contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(thresh, contours, -1, (0,255,0), 3)
#print(contours)


cv2.imshow('image', img)
cv2.imshow('gray', gray)
cv2.imshow('thresh', thresh)
cv2.imshow('blur',blur)
cv2.imshow('morph',morph)
cv2.waitKey(0)
cv2.destroyAllWindows()
