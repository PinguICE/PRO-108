import cv2
import mediapipe as mp
import math
from pynput.mouse import Button, Controller
import pyautogui
mouse = Controller()

catnap = cv2.VideoCapture(0)

width = int(catnap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(catnap.get(cv2.CAP_PROP_FRAME_HEIGHT))
(screen_width,screen_height) = pyautogui.size()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence = 0.8,min_tracking_confidence = 0.5)
tipID = [4,8,12,16,20]

def count_finger(image,handpoints,hand_num = 0):
    global pinch

    if handpoints:
        points = handpoints[hand_num].landmark
        fingers = []

        for img_index in tipID:
            finger_tip_y = points[img_index].y
            finger_button_y = points[img_index - 2].y
            if img_index != 4:
                if finger_tip_y < finger_button_y:
                    fingers.append(1)
            

                if finger_tip_y > finger_button_y:
                    fingers.append(0)

        totalfingers = fingers.count(1)
        text = f'fingers:{totalfingers}'
        cv2.putText(image,text,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

        finger_tip_x = int((handpoints[8].x)*width)
        finger_tip_y = int((handpoints[8].y)*height)

        thumb_tip_x = int((handpoints[4].x)*width)
        thumb_tip_y = int((handpoints[4].y)*height)

        cv2.line(image,(finger_tip_x,finger_tip_y),(thumb_tip_x,thumb_tip_y),(255,0,0),2)
        center_x = int((finger_tip_x + thumb_tip_x)/2)
        center_y = int((finger_tip_y + thumb_tip_y)/2)
        cv2.circle(image,(center_x, center_y),(255,0,0),2)

        distance = math.sqrt(((finger_tip_x - thumb_tip_x)**2)+((finger_tip_y - thumb_tip_y)**2))
        relative_mouse_x = (center_x / width)*screen_width
        relative_mouse_y = (center_y / width)*screen_height
        mouse.position = (relative_mouse_x,relative_mouse_y)


def draw_points(image,handspoint):
    if handspoint:
        for points in handspoint:
            mp_drawing.draw_landmarks(image,points,mp_hands.HAND_CONNECTIONS)
            
while True:
    success, image = catnap.read()
    image = cv2.flip(image,1)
    results = hands.process(image)
    handspoint = results.multi_hand_landmarks
    draw_points(image,handspoint)
    count_finger(image,handspoint)
    cv2.imshow("Leyendo img",image)
    key = cv2.waitKey(1)
    if key == 32:
        break
cv2.destroyAllWindows()
 
# para copiar python3 gui.py