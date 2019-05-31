import numpy as np
import cv2
from keras.models import load_model

#start capturing the video
cap = cv2.VideoCapture(0)

#dimensions of the region of interest
x = 50
y = 60
w = 200
h = 200

#get weights from the trained model
model = load_model('model.h5')

#this function is used to predict the image
def predictionImage(roi,thresh):
	img = np.zeros_like(roi,np.float32)

	#converting 1 channel threshold image to 3 channel image for our model
	img[:,:,0] = thresh
	img[:,:,1] = thresh
	img[:,:,2] = thresh
	img = img.reshape(1,200,200,3)
	#normalizing the image
	img /= 255.

	return img

#preprocessing as done while creating the dataset
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

	#get image to be used for prediciton
	img = predictionImage(roi,thresh)
	

	return mask,thresh,img

#write the predicted text
def writeTextToWindow(img,text,default_x_calc,default_y_calc):
	fontscale = 1.0
	color = (0, 0, 0)
	fontface = cv2.FONT_HERSHEY_COMPLEX_SMALL
	cv2.putText(img, str(text), (default_x_calc, default_y_calc), fontface, fontscale, color)
	

	return img

#this array contains the first and the second operand that is to be used in calculation
predArray = [-1,-1]

#dimensions used while writing the predicted text
default_y_calc = 80	
default_x_calc = 25


predCount = 0 #for confirming the number displayed
predPrev = 0

#space for writing the predicted text
result = np.zeros((300,300,3),np.uint8)
result.fill(255) #fill result window(make it white)
cv2.putText(result,"Calculator", (25, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0,0,0))


while True:
	ret,frame = cap.read() #read frame
	
	mask,thresh,img = imagePreprocess(frame)

	

	#if we get the same prediction for 15 times, we take it as the confirmed prediction
	if predCount > 15:
		print('Prediction: ' + str(predPrev))
		
		#check whether it is the first operand or the second
		if predArray[0] == -1: 
			predArray[0] = predPrev
			string = '{} + '.format(predArray[0])
			writeTextToWindow(result,string,default_x_calc,default_y_calc)
			default_x_calc += 20

		else:
			default_x_calc += 40
			predArray[1] = predPrev
			string = '{} = {}'.format(predArray[1],np.sum(predArray,axis=0))
			writeTextToWindow(result,string,default_x_calc,default_y_calc)

			default_x_calc = 25
			default_y_calc += 30

			print("Sum: {}".format(np.sum(predArray,axis=0)))
			predArray = [-1,-1] #reset the values of the operands
		predCount = 0 #start counting again to get the next prediction

	
	predict = model.predict(img) #predict the number
	pred = predict.argmax() 
	#increase predCount only if the previous prediction matches with our current prediction
	if predPrev == pred:
		predCount+=1
	else:
		predCount = 0

	predPrev = pred
	
	
	#showing the required windows
	cv2.imshow("result",result) #window for prediciton
	cv2.imshow('frame',frame) #main webcam window
	#cv2.imshow('roi',mask)
	cv2.imshow('thresh',thresh) #window to show the thresholded image that is being used for prediction


	k = cv2.waitKey(30) & 0xff #exit if Esc is pressed
	if k == 27:
		break

cap.release() #release the webcam
cv2.destroyAllWindows() #destroy the window