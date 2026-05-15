import cv2
cap = cv2.VideoCapture(1)
status , photo = cap.read()
print(status)
cv2.imshow("hi",photo )
cv2.waitKey(10000)
cv2.destroyAllWindow()

from cvzone.handtrackingModule import HandDetector
brain_hand_detector = HandDetector()
my_hand_detector = brain_hand_detector.findHands(photo)

my_hand_lmlist = my_hand_detector[0][0]
my_finger_up = brain_hand_detector.fingersUp(my_hand_lmlist)

# import os

# if my_finger_up ==[1 ,1,1,1,1]:
#     os.system("notepad")
# elif my_finger_up ==[0,1,1,0,0]:
#     os.system("chrome")
# else:
#     print("idk")