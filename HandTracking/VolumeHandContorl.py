# hand detection modules
import cv2
import HandTrackingModule as htm

# misc modules
import numpy as np
import math
import time
import matplotlib
from matplotlib import cm

# pycaw modules
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)  # setting camera width to wCam
cap.set(4, hCam)  # setting camera height to hCam

minLength = 30  # min length b/w fingers, depending on user
maxLength = 120  # max length b/w fingers, depending on user

cmap = matplotlib.cm.get_cmap('hot')  # getting cmap
volCmapB, volCmapG, volCmapR = 0, 0, 0

# initialising pycaw
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol, volBar, volPercent = 0, 400, 0

pTime = 0

detector = htm.handDetector(maxHands=1, detectionCon=.8)  # setting up hand model
while True:
    success, img = cap.read()

    img = detector.findHands(img, draw=False)  # find hands
    lmList = detector.findPosition(img, draw=False)  # getting location of hand landmarks
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # getting center of the 2 points using midpoint formula

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # making a line b/w the 2 points

        #       horizontal distance, vertical distance
        length = math.hypot(x2 - x1, y2 - y1)  # getting distance between the 2 points

        # hand range: [30, 120], [minLength, maxLength]
        # vol range: [-63, 0], [minVol, maxVol]
        # color values for cv2 shapes (blue, green, red): [0, 255]
        # cmap domain: [0, 255] int values
        # cmap range: [0, 1]
        # volBar: [400, 148] y-axis values of the bar

        vol = np.interp(length, [minLength, maxLength], [minVol, maxVol])  # length to volume range
        # volume.SetMasterVolumeLevel(vol, None)  # setting volume

        volColor = np.interp(length, [minLength, maxLength], [0, 255])  # length to color range
        rgba = cmap(int(volColor))  # getting cmap color (rgba) of that color
        volCmapR = np.interp(rgba[0], [0.0, 1.0], [0, 255])  # cmap to color for red
        volCmapG = np.interp(rgba[1], [0.0, 1.0], [0, 255])  # cmap to color for green
        volCmapB = np.interp(rgba[2], [0.0, 1.0], [0, 255])  # cmap to color for blue

        volBar = np.interp(length, [minLength, maxLength], [400, 148])  # length to height of volBar
        volPercent = np.interp(length, [minLength, maxLength], [0, 100])  # length to percent
        cv2.circle(img, (cx, cy), 10, (volCmapB, volCmapG, volCmapR), cv2.FILLED)  # making a circle at the midpoint

        # cv2.putText(img, str(int(length)), (10, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        # cv2.putText(img, str(int(vol)), (100, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cv2.rectangle(img, (50, 148), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (volCmapB, volCmapG, volCmapR), cv2.FILLED)
    cv2.putText(img, f"{int(volPercent)}%", (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (10, 35), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
