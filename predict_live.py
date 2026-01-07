# suppressing known warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


import cv2
import mediapipe as mp
import math
import joblib

model = joblib.load("knn_model.pkl")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    predict_label = "None"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark

            wrist = landmarks[0]
            ref = landmarks[9]

            scale = math.sqrt(
                (ref.x - wrist.x) ** 2 +
                (ref.y - wrist.y) ** 2 +
                (ref.z - wrist.z) ** 2
            )

            if scale == 0:
                continue

            features = []
            for lm in landmarks:
                features.extend([
                    (lm.x - wrist.x) / scale,
                    (lm.y - wrist.y) / scale,
                    (lm.z - wrist.z) / scale
                ])

            prediction = model.predict([features])
            predict_label = prediction[0]
            cv2.putText(
                frame,
                f"Prediction: {predict_label}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),

            )

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
hands.close()
cv2.destroyAllWindows()


