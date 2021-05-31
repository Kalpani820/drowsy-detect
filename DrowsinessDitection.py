#Driver Drowsiness Detection System Code in Python

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
import RPi.GPIO as GPIO
from time import sleep

#Select GPIO mode
GPIO.setmode(GPIO.BCM)
#Set buzzer - pin 23 as output
buzzer=23 
GPIO.setup(buzzer,GPIO.OUT)


#-------------------------- Necessary Functions--------------------------------

#-----Play an alarm sound------
def WakeupAlarm():
    global status
    
    while status:
        GPIO.output(buzzer,GPIO.HIGH)
        sleep(0.5)
        GPIO.output(buzzer,GPIO.LOW)
        sleep(0.5)

        
    
#Compute the ratio of distances between the vertical eye landmarks and the distances between the horizontal eye landmarks
def eyeAspectRaio(eye):
    # compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C) # compute the eye aspect ratio

    return ear 

# grab the indexes of the facial landmarks for the left and right eye, respectively
def finalEar(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eyeAspectRaio(leftEye)
    rightEAR = eyeAspectRaio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)


def lipDistance(shape):
    upperLip = shape[50:53]
    upperLip = np.concatenate((upperLip, shape[61:64]))

    lowerLip = shape[56:59]
    lowerLip = np.concatenate((lowerLip, shape[65:68]))

    topMean = np.mean(upperLip, axis=0)
    lowMean = np.mean(lowerLip, axis=0)

    distance = abs(topMean[1] - lowMean[1])
    return distance

#-----------------main---------------------------------

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,help="index of webcam on system")
args = vars(ap.parse_args())

EYE_THRESH = 0.3
EYE_CONSEC_FRAMES = 30 #number of frames which want to check blink or not
YAWN_THRESH = 20
status=False
COUNTER = 0

#detect face
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    #Faster but less accurate (dlib library is less faster)
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') #detect land marks


print("Loading........")

vs= VideoStream(usePiCamera=True).start()       
time.sleep(1.0)

while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=450) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect face from gray scale image
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)  

       
    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
        
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = finalEar(shape)
        ear = eye[0]
        leftEye = eye [1]
        rightEye = eye[2]

        distance = lipDistance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        # check to see if the eye aspect ratio is below the blink threshold, and if so, increment the blink frame counter
        if ear < EYE_THRESH:
            COUNTER += 1

            
            if COUNTER >= EYE_CONSEC_FRAMES:
                if status == False:
                    status = True
                    t = Thread(target=WakeupAlarm)
                    t.deamon = True
                    t.start()

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
        else:
            COUNTER = 0
            status = False

        if (distance > YAWN_THRESH):
                cv2.putText(frame, "Yawn Alert", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if status == False :
                    status= True
                    t = Thread(target=WakeupAlarm)
                    t.deamon = True
                    t.start()
                                        
        else:
            status = False
            
        
        #printing values
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    
    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()