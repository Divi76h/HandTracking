import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)  # choosing webcam

# formalities for mediapipe's hand model and drawing_utils
mpHands = mp.solutions.hands
hands = mpHands.Hands()
"""
parameters for hand model
static_image_mode=False: track and detect based on confidence. True: always detect, make it slow
max_num_hands=2: 
model_complexity=1: # new parameter, better not touch it
min_detection_confidence=0.5: 50%
min_tracking_confidence=0.5): 50%
"""
mpDraw = mp.solutions.drawing_utils

pTime = 0  # previous time

while True:
    success, img = cap.read()  # getting data from webcam
    imgRBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converting to rgb form for hand model
    results = hands.process(imgRBG)  # asking handel model to process
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:  # if the hand model gets detects a landmark
        for handLms in results.multi_hand_landmarks:  # iterating over all landmarks
            for id, lm in enumerate(handLms.landmark):  # getting if and landmark's position
                h, w, c = img.shape  # getting shape of camera
                cx, cy = int(lm.x * w), int(lm.y * h)  # getting pixel values for the landmark's position

                if id == 8:  # if index finger, draw circle on it
                    #        where, position, radius, color,     filled
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            # using drawing_utils to draw landmarks and lines b/w them
            #                   where, landmarks, connections
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # calculating and displaying fps
    cTime = time.time()  # current time from time module
    fps = 1 / (cTime - pTime)  # 1 / time between each frame = fps
    pTime = cTime
    #         where, text,          position, font,               scale, color,         thickness
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # showing webcam
    cv2.imshow('Image', img)
    cv2.waitKey(1)
