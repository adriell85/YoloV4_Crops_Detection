import cv2
import numpy as np

img = cv2.imread('placa_torta/plate_4.png', 1)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)
imgCanny = cv2.Canny(imgBlur, 150, 200)
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rectCon = []
for cont in contours:
    area = cv2.contourArea(cont)
    if area > 100:
        # print(area) #prints all the area of the contours
        peri = cv2.arcLength(cont, True)
        approx = cv2.approxPolyDP(cont, 0.01 * peri, True)
        # print(len(approx)) #prints the how many corner points does the contours have
        if len(approx) == 4:
            rectCon.append(cont)
            # print(len(rectCon))

rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)  # Sort out the contours based on largest area to smallest

bigPeri = cv2.arcLength(rectCon[0], True)
cornerPoints = cv2.approxPolyDP(rectCon[0], 0.01 * peri, True)

# Reorder bigCornerPoints so I can prepare it for warp transform (bird eyes view)
cornerPoints = cornerPoints.reshape((4, 2))
mynewpoints = np.zeros((4, 1, 2), np.int32)
add = cornerPoints.sum(1)

mynewpoints[0] = cornerPoints[np.argmin(add)]
mynewpoints[3] = cornerPoints[np.argmax(add)]
diff = np.diff(cornerPoints, axis=1)
mynewpoints[1] = cornerPoints[np.argmin(diff)]
mynewpoints[2] = cornerPoints[np.argmax(diff)]

# Draw my corner points
# cv2.drawContours(img,mynewpoints,-1,(0,0,255),10)

##cv2.imshow('Corner Points in Red',img)
##print(mynewpoints)

# Bird Eye view of your region of interest
pt1 = np.float32(mynewpoints)  # What are your corner points
pt2 = np.float32([[0, 0], [300, 0], [0, 200], [300, 200]])
matrix = cv2.getPerspectiveTransform(pt1, pt2)
imgWarpPers = cv2.warpPerspective(img, matrix, (300, 200))
cv2.imshow('Result', imgWarpPers)