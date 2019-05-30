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
	blur = cv2.GaussianBlur(mask,(5,5),1)
	dilation = cv2.dilate(blur,kernel,iterations = 1)
	erosion = cv2.erode(dilation,kernel,iterations=1)
	ret,thresh = cv2.threshold(erosion,127,255,0)

	return mask,thresh


count = 0
#ready = False
frameCount = 0
while True:
	ret,frame = cap.read()
	imgName = str(count) + '.jpg'
	frameCount+=1
	# if ready == False:
	# 	time.sleep(1)
	# 	ready = True
	
	mask,thresh = imagePreprocess(frame)

	if frameCount%5 == 0:
		if count<1200:
			cv2.imwrite(imgName,thresh)
			count+=1
	

	cv2.imshow('frame',frame)
	cv2.imshow('roi',mask)
	cv2.imshow('thresh',thresh)


	k = cv2.waitKey(30) & 0xff #exit if Esc is pressed
	if k == 27:
		break



cap.release() #release the webcam
cv2.destroyAllWindows() #destroy the window