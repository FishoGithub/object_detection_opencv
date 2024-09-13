import cv2
import PyObjCTools
from playsound import playsound
import pygame
import simpleaudio

# img = cv2.imread('no_arrow.png')
# img2 = cv2.imread('no_arrow.png')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

classNames = []

classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
pygame.mixer.init()


# pygame.mixer.music.load('Northwest Benefit Auctions.mp3')
# pygame.mixer.music.play()

def play():
    wave_obj = simpleaudio.WaveObject.from_wave_file('Northwest Benefit Auctions.mp3')
        # Play the audio file
    play_obj = wave_obj.play()
        # Wait for the audio to finish playing
    play_obj.wait_done()


while True:
    success, img2 = cap.read()
    # classIds, confs, bbox = net.detect(img, confThreshold=0.35)  #arrow pic
    classIds, confs, bbox = net.detect(img2, confThreshold=0.5)  #no arrow pic
    print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img2, box, color=(0, 0, 255), thickness=2)
            cv2.putText(img2, classNames[classId - 1], (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_PLAIN, 2,
                        (0, 255, 0), 2)
            if classId == 35 or classId == 34 or classId == 10:
                # playsound('Northwest Benefit Auctions.mp3', True)
                cv2.putText(img2, "ARROW DETECTED", (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0),
                            2)
                break

    cv2.imshow("output", img2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyWindow()
        break


