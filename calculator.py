import numpy as np
import cv2
from keras.models import load_model

cap = cv2.VideoCapture(0)

x = 50
y = 60
w = 200
h = 200

#get weights from the trained model
model = load_model('model.h5')

def predictionImage(roi,thresh):
	img = np.zeros_like(roi)
	img[:,:,0] = thresh
	img[:,:,1] = thresh
	img[:,:,2] = thresh
	img = img.reshape(1,200,200,3)

	return img

def imagePreprocess(frame):

	cv2.rectangle(frame,(x,y),(w+x,h+y),(0,255,0),2)
	roi = frame[y:h+y,x:w+x]

	hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
	#mask for thresholding the skin color
	mask = cv2.inRange(hsv,np.array([2,30,50]),np.array([30,255,255]))

	#reducing the noise in the image
	kernel = np.ones((5,5))
	blur = cv2.GaussianBlur(mask,(5,5),1)
	dilation = cv2.dilate(blur,kernel,iterations = 1)
	erosion = cv2.erode(dilation,kernel,iterations=1)
	ret,thresh = cv2.threshold(erosion,127,255,0)


	img = predictionImage(roi,thresh)
	

	return mask,thresh,img
	


predCount = 0 #for confirming the number displayed
predPrev = 0
while True:
	ret,frame = cap.read()
	
	mask,thresh,img = imagePreprocess(frame)

	


	if predCount > 15:
		print('Prediction: ' + str(predPrev))
		#putTextInResult(blackboard, predPrev)
		predCount = 0

	
	predict = model.predict(img)
	pred = predict.argmax()
	if predPrev == pred:
		predCount+=1
	else:
		predCount = 0

	predPrev = pred
	
	#print(pred)


	#result = np.zeros((200, 200, 3), dtype=np.uint8)
	#res = np.hstack((result, blackboard))
	#cv2.imshow("Recognizing gesture", res)
	cv2.imshow('frame',frame)
	cv2.imshow('roi',mask)
	cv2.imshow('thresh',thresh)


	k = cv2.waitKey(30) & 0xff #exit if Esc is pressed
	if k == 27:
		break

cap.release() #release the webcam
cv2.destroyAllWindows() #destroy the window