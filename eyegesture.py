import cv2
import mediapipe as mp
import pyautogui
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame_height, frame_width, _ = image.shape  # Get frame dimensions

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Use refined eye landmarks for better accuracy
                left_eye_indices = [362, 385, 387, 263, 373, 380]  # Right eye (mirror)
                right_eye_indices = [33, 160, 158, 133, 153, 144]  # Left eye (mirror)

                left_eye_x = np.mean([face_landmarks.landmark[i].x for i in left_eye_indices]) * frame_width
                left_eye_y = np.mean([face_landmarks.landmark[i].y for i in left_eye_indices]) * frame_height

                right_eye_x = np.mean([face_landmarks.landmark[i].x for i in right_eye_indices]) * frame_width
                right_eye_y = np.mean([face_landmarks.landmark[i].y for i in right_eye_indices]) * frame_height

                # Compute average eye position
                eye_x = int((left_eye_x + right_eye_x) / 2)
                eye_y = int((left_eye_y + right_eye_y) / 2)

                # Scale to screen dimensions
                screen_x = int(eye_x / frame_width * screen_width)
                screen_y = int(eye_y / frame_height * screen_height)

                # Move the cursor
                pyautogui.moveTo(screen_x, screen_y, duration=0.1)  # Smooth movement

                # Draw facial landmarks
                mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

        cv2.imshow('Eye Mouse Control', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
