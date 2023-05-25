import cv2
import autopy
import mediapipe as mp
import math


def smoothing_factor(cutoff_frequency, sampling_period):
    r = 2 * math.pi * cutoff_frequency * sampling_period
    return r / (r + 1)


def exponential_smoothing(alpha, signal, filtered_signal):
    return alpha * signal + (1 - alpha) * filtered_signal


def one_euro_filter(signal, prev_filtered_signal, prev_filtered_rate_of_change, minimum_cutoff_frequency, beta):
    default_cutoff_frequency = 1
    sampling_period = 0.001
    alpha_d = smoothing_factor(default_cutoff_frequency, sampling_period)
    rate_of_change = (signal - prev_filtered_signal) / sampling_period
    filtered_rate_of_change = exponential_smoothing(alpha_d, rate_of_change, prev_filtered_rate_of_change)
    cutoff_frequency = minimum_cutoff_frequency + beta * abs(filtered_rate_of_change)
    alpha = smoothing_factor(cutoff_frequency, sampling_period)
    filtered_signal = exponential_smoothing(alpha, signal, prev_filtered_signal)
    return filtered_signal, filtered_rate_of_change


cap = cv2.VideoCapture(0)
width, height = autopy.screen.size()
hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_tracking_confidence=0.5,
                                 min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
f1, f2 = False, False
start = False
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
                    if start:
                        cx_filtered, prev_x_rate_of_change = one_euro_filter(cx, prev_cx_filtered,
                                                                             prev_x_rate_of_change, 0.000000000000001,
                                                                             100)
                        cy_filtered, prev_y_rate_of_change = one_euro_filter(cy, prev_cy_filtered,
                                                                             prev_y_rate_of_change, 0.000000000000001,
                                                                             100)
                    else:
                        cx_filtered, cy_filtered = cx, cy
                        prev_x_rate_of_change, prev_y_rate_of_change = 0, 0
                        start = True
                    autopy.mouse.move(cx_filtered * width / w, cy_filtered * height / h)
                    print(cx_filtered, cy_filtered, sep=' ', end='; ')
                    prev_cx_filtered, prev_cy_filtered = cx_filtered, cy_filtered
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
