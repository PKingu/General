import cv2
import mediapipe                                         # Import Libraries
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(-1)                               # Webcam Setup
cap.set(3, 1280)
cap.set(4, 720)


while True:
    success, img = cap.read(0)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
