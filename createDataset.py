import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)

x = 50
y = 60
w = 200
h = 200

def imagePreprocess(frame):

	cv2.rectangle(frame,(x,y),(w+x,h+y),(0,255,0),2)
	roi = frame[y:h+y,x:w+x]

	hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
	#mask for thresholding the skin color
	mask = cv2.inRange(hsv,np.array([2,20,50]),np.array([30,255,255]))

	#reducing the noise in the image
	kernel = np.ones((5,5))
	#applying gaussian blur to reduce noise
	blur = cv2.GaussianBlur(mask,(5,5),1)

	#applying morphological operations 
	#dilation for filling the gaps in the image
	#erosion for thinning the image obtained after dilation
	dilation = cv2.dilate(blur,kernel,iterations = 1)
	erosion = cv2.erode(dilation,kernel,iterations=1)
	#thresholding the image
	ret,thresh = cv2.threshold(erosion,127,255,0)

	return mask,thresh


count = 0
#ready = False
frameCount = 0
while True:
	#capture frame
	ret,frame = cap.read()
	imgName = str(count) + '.jpg' #name of the image
	frameCount+=1
	# if ready == False:
	# 	time.sleep(1)
	# 	ready = True
	
	mask,thresh = imagePreprocess(frame)

	if frameCount%5 == 0: #if frame count is a multiple of 5, read that frame and write it into the given file location
		if count<1200: #capture 1200 images
			cv2.imwrite(imgName,thresh)
			count+=1
	

	#showing the required frames
	cv2.imshow('frame',frame)
	cv2.imshow('roi',mask)
	cv2.imshow('thresh',thresh)


	k = cv2.waitKey(30) & 0xff #exit if Esc is pressed
	if k == 27:
		break



cap.release() #release the webcam
cv2.destroyAllWindows() #destroy the window