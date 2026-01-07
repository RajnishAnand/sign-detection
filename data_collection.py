# suppressing known warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import mediapipe as mp
import math
import csv
import os


 # Change this label as needed for different signs
current_label =  input("Enter current_label for data collection: ")
if(len(current_label) > 1): 
    raise RuntimeError("incorrect label, length exceeds 1.")
current_label = current_label.upper()

DATASET_PATH = "./sign_dataset.csv"
if not os.path.exists(DATASET_PATH):
    with open(DATASET_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["label"] + [f"f{i}" for i in range(63)])


# hand traking modules and functions
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
#

hands = mp_hands.Hands(
   static_image_mode=False,
   max_num_hands=1,
   min_detection_confidence=0.7,
   min_tracking_confidence=0.7
)

# req for camera
cap = cv2.VideoCapture(0)

# camera check 
if not cap.isOpened():
    raise RuntimeError("Camera could not be opened")

sample_counter = 0
# display camera feed
# ret → boolean success flag
# frame → image matrix (NumPy array)
while True:
    ret, frame = cap.read()

    if not ret:
        print ("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hands_landmarks in results.multi_hand_landmarks:

            landmarks = hands_landmarks.landmark
            wrist = landmarks[0]
            ref = landmarks[9]

            scale = math.sqrt(
              (ref.x - wrist.x) ** 2 +
              (ref.y - wrist.y) ** 2 +
              (ref.z - wrist.z) ** 2
            )

            if scale == 0: continue
            # print(f'Wrist coordinates: (x={wrist.x:3f}, y={wrist.y:.3f}, z={wrist.z:.3f}), scale={scale:.3f}', end="\r")

            features = []
            for lm in landmarks:
                features.extend([
                    (lm.x - wrist.x) / scale,
                    (lm.y - wrist.y) / scale,
                    (lm.z - wrist.z) / scale
                ])

            # print(f"Features sample: {features[:6][0]:3f} {features[:6][1]:3f} {features[:6][2]:3f}", end="\r")

            mp_drawing.draw_landmarks(
                frame,
                hands_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
            )


    # vid to gui window
    cv2.putText(frame, f'Label: {current_label}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.imshow("Hand Landmrks", frame)

    key = cv2.waitKey(1) & 0xFF
    # exit control
    if  key == ord('q'):
        break
    elif key == ord('s'):
        with open(DATASET_PATH, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([current_label] + features)
        sample_counter+=1
        print(f"Saved data +{sample_counter} for label: {current_label}", end="\r")

# returns camera to os
cap.release()
cv2.destroyAllWindows()
