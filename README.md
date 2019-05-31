# Gesture Calculator

<p>Gesture Calculator is a <b>gesture recognition</b> program which can detect counting from 0-9 using hand gestures.
After detecting 2 hand gestures, it calculates the sum of the predicted gestures and displays the output.</br>
Gesture Recognition was done with the help of <b>CNN algorithm</b>, which is a great algorithm for image related problems.</br>
The dataset was created with the help of <b>openCV</b> library by capturing frames from the webcam.
</p>

## Prerequisites


**Libraries Required:**
- tensorflow 1.3.0
- keras 2.2.4
- numpy 1.13.1
- opencv-contrib-python 4.0.0.21

## Working

1. First we create a dataset for our classifier. We capture 1200 images for each class. The image is preprocessed(Thresholding,masking,etc). Later data augmentation is done to increase the data
2. Train the CNN algorithm on the created dataset. A custom CNN model was created in this project. Transfer learning can also be used. Save the weights obtained after training
3. The weights are used to classify the gestures in real time. When 2 gestures are obtained, the sum is displayed

## Demo

<img src="https://github.com/vibhusehra/Gesture-Calculator/blob/master/Results/record.gif" width="500" height="500" />

Nine                       +  Eight
:-------------------------:|:-------------------------:
![](https://github.com/vibhusehra/Gesture-Calculator/blob/master/Results/nine.PNG)  |  ![](https://github.com/vibhusehra/Gesture-Calculator/blob/master/Results/eight.PNG)

#### Validation Accuracy
```
99%
```
