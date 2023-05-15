import cv2
import autopy
import mediapipe as mp
cap = cv2.VideoCapture(0)
width, height = autopy.screen.size()
hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_tracking_confidence=0.5,
                                 min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
f1, f2 = False, False
while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    result = hands.process(img)
    if f1 and not f2:
        print('\nHand disappeared')
    if result.multi_hand_landmarks:
        for id_finger, lm in enumerate(result.multi_hand_landmarks[0].landmark):
            h, w, _ = img.shape
            f1 = f2
            f2 = True
            if not f1 and f2:
                print('Hand appeared')
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 3, (255, 0, 255))
            if id_finger == 4:
                cx_2, cy_2 = cx, cy
            if id_finger == 8:
                cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
                cx_1, cy_1 = cx, cy
                try:
                    autopy.mouse.move(cx * width / w, cy * height / h)
                    print(cx, cy, sep=' ', end='; ')
                except ValueError:
                    continue
                if ((cx_1 - cx_2) ** 2 + (cy_1 - cy_2) ** 2) ** 0.5 < 50:
                    try:
                        autopy.mouse.click()
                    except ValueError:
                        continue
        mpDraw.draw_landmarks(img, result.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)
    else:
        f1 = f2
        f2 = False
    cv2.imshow("Hand tracking", img)
    cv2.waitKey(1)
