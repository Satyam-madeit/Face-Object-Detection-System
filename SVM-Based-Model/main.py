import pickle
import cv2

with open(r'SVM-Based-Model\trained_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
       print("Error: Failed to grab frame.")
       break
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break