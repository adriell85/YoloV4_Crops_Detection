import cv2
import numpy as np
from matplotlib import pyplot as plt

def order_points(pts):
    # Step 1: Find centre of object
    center = np.mean(pts)

    # Step 2: Move coordinate system to centre of object
    shifted = pts - center

    # Step #3: Find angles subtended from centroid to each corner point
    theta = np.arctan2(shifted[:, 0], shifted[:, 1])

    # Step #4: Return vertices ordered by theta
    ind = np.argsort(theta)
    return pts[ind]

def getContours(img, orig):  # Change - pass the original image too
    biggest = np.array([])
    maxArea = 0
    imgContour = orig.copy()  # Make a copy of the original image to return
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    index = None
    for i, cnt in enumerate(contours):  # Change - also provide index
        area = cv2.contourArea(cnt)
        if area > 500:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt,0.02*peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
                index = i  # Also save index to contour

    warped = None  # Stores the warped license plate image
    if index is not None: # Draw the biggest contour on the image
        cv2.drawContours(imgContour, contours, index, (255, 0, 0), 3)

        src = np.squeeze(biggest).astype(np.float32) # Source points
        height = image.shape[0]
        width = image.shape[1]
        # Destination points
        dst = np.float32([[0, 0], [0, height - 1], [width - 1, 0], [width - 1, height - 1]])

        # Order the points correctly
        biggest = order_points(src)
        dst = order_points(dst)

        # Get the perspective transform
        M = cv2.getPerspectiveTransform(src, dst)

        # Warp the image
        img_shape = (width, height)
        warped = cv2.warpPerspective(orig, M, img_shape, flags=cv2.INTER_LINEAR)

    return biggest, imgContour, warped  # Change - also return drawn image

kernel = np.ones((3,3))
image = cv2.imread('placa_torta/plate_7.png')




imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)

thresh = np.uint8(np.where(imgBlur>90,255,0))

imgCanny = cv2.Canny(imgBlur,150,200)
imgDial = cv2.dilate(imgCanny,kernel,iterations=2)
imgThres = cv2.erode(imgDial,kernel,iterations=2)
biggest, imgContour, warped = getContours(imgThres, image)  # Change

cv2.imshow('Original',image)
cv2.imshow('gray',imgGray)
cv2.imshow('adaptative_threshold',thresh)
# cv2.imshow('Blur',imgBlur)
cv2.imshow('Canny',imgCanny)
# cv2.imshow('Dilate',imgDial)
# cv2.imshow('Threshold',imgThres)
# cv2.imshow('Contours',imgContour)
try:
    cv2.imshow('Warped',warped)
except:
    None
cv2.waitKey(0)
# titles = ['Original', 'Blur', 'Canny', 'Dilate', 'Threshold', 'Contours', '']  # Change - also show warped image
# images = [image[...,::-1],  imgBlur, imgCanny, imgDial, imgThres, imgContour, warped]  # Change
#
# # Change - Also show contour drawn image + warped image
# for i in range(5):
#     plt.subplot(3, 3, i+1)
#     plt.imshow(images[i], cmap='gray')
#     plt.title(titles[i])
#
# plt.subplot(3, 3, 6)
# plt.imshow(images[-2])
# plt.title(titles[-2])
#
# plt.subplot(3, 3, 8)
# plt.imshow(images[-1])
# plt.title(titles[-1])
#
# plt.show()