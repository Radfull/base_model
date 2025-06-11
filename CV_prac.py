import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np

mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

YOUR_NAME = "Radmir"
YOUR_SURNAME = "Kalimullin"
KNOWN_FACE_ENCODINGS = []  

def count_fingers(hand_landmarks):
    """Считаем поднятые пальцы"""
    tip_ids = [4, 8, 12, 16, 20] 
    fingers = []
    
    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0]-1].x:
        fingers.append(1)
    else:
        fingers.append(0)
    
    for id in range(1, 5):
        if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id]-2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    
    return sum(fingers)

def main():
    cap = cv2.VideoCapture(0)
    
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection, \
        mp_hands.Hands(
            max_num_hands=1, min_detection_confidence=0.7) as hands:
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            hand_results = hands.process(rgb_frame)
            fingers_count = 0
            
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    fingers_count = count_fingers(hand_landmarks)
            
            face_results = face_detection.process(rgb_frame)
            emotion = ""
            
            if face_results.detections:
                for detection in face_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                 int(bboxC.width * iw), int(bboxC.height * ih)
                    
                    x = max(0, x - 20)
                    y = max(0, y - 20)
                    w = min(iw - x, w + 40)
                    h = min(ih - y, h + 40)
                    
                    face_img = frame[y:y+h, x:x+w]
                    
                    if fingers_count == 3:
                        try:
                            analysis = DeepFace.analyze(
                                img_path=face_img, 
                                actions=['emotion'],
                                enforce_detection=False
                            )
                            if isinstance(analysis, list):
                                analysis = analysis[0]
                            emotion = analysis['dominant_emotion']
                        except:
                            emotion = "unknown"
                    
                    color = (0, 255, 0) 
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    text = "unknown"
                    if fingers_count == 1:
                        text = YOUR_NAME
                    elif fingers_count == 2:
                        text = YOUR_SURNAME
                    elif fingers_count == 3:
                        text = emotion if emotion else "unknown"
                    
                    cv2.putText(frame, text, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            cv2.putText(frame, f"Fingers: {fingers_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Face and Hand Detection', frame)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()