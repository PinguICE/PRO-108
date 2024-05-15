import cv2
import mediapipe as mp
catnap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence = 0.8,min_tracking_confidence = 0.5)
tipID = [4,8,12,16,20]

def count_finger(image,handpoints,hand_num = 0):
    if handpoints:
        points = handpoints[hand_num].landmark
        fingers = []

        for img_index in tipID:
            finger_tip_y = points[img_index].y
            finger_button_y = points[img_index - 2].y
            if img_index != 4:
                if finger_tip_y < finger_button_y:
                    fingers.append(1)
                    print("El dedo con ID",img_index,"Está abierto")

                if finger_tip_y > finger_button_y:
                    fingers.append(1)
                    print("El dedo con ID",img_index,"Está cerrado")

        totalfingers = fingers.count(1)
        text = f'fingers:{totalfingers}'
        cv2.putText(image,text,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)


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
 
# para copiar python3 rename.py