import cv2
import mediapipe as mp
import pyautogui
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

screen_width, screen_height = pyautogui.size()

with mp_face_mesh.FaceMesh(max_num_faces=1) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get the coordinates of the left and right eye landmarks
                left_eye_landmarks = [face_landmarks.landmark[i] for i in range(133, 144)]
                right_eye_landmarks = [face_landmarks.landmark[i] for i in range(362, 373)]

                # Calculate the average position of the left and right eye landmarks
                left_eye_x = int(np.mean([landmark.x for landmark in left_eye_landmarks]) * screen_width)
                left_eye_y = int(np.mean([landmark.y for landmark in left_eye_landmarks]) * screen_height)
                right_eye_x = int(np.mean([landmark.x for landmark in right_eye_landmarks]) * screen_width)
                right_eye_y = int(np.mean([landmark.y for landmark in right_eye_landmarks]) * screen_height)

                eye_x = (left_eye_x + right_eye_x) // 2
                eye_y = (left_eye_y + right_eye_y) // 2

                pyautogui.moveTo(eye_y, eye_x)

                mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

        cv2.imshow('Eye Mouse Control', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()