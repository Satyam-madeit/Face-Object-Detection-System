import pickle
import cv2
from skimage.feature import hog

def img_pyramid(image):
   yield image
   while True:
       image = cv2.pyrDown(image)
       if image.shape[0] < 64 or image.shape[1] < 64:
           break
       yield image

def sliding_window(img, step_size):
    for y in range(0, img.shape[0], step_size):
         for x in range(0, img.shape[1], step_size):
              patch = img[y:y+64, x:x+64]
              if patch.shape[0] != 64 or patch.shape[1] != 64:
                   continue
              yield (x, y, patch)

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
    detections = []
    for resized in img_pyramid(frame):
        for (x, y, patch) in sliding_window(resized, step_size=64):
            features = hog(patch, orientations=8, 
                pixels_per_cell=(8, 8), 
                cells_per_block=(1, 1), 
                visualize=False, 
                channel_axis=-1)
            features = features.reshape(1, -1)
            prediction = loaded_model.predict(features)
            if prediction == 1:
                scale_x = frame.shape[1] / resized.shape[1]
                scale_y = frame.shape[0] / resized.shape[0]
                detections.append((int(x * scale_x), int(y * scale_y), int(64 * scale_x), int(64 * scale_y)))
    rects, weights = cv2.groupRectangles(detections + detections, groupThreshold = 4, eps=0.3)  
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)     
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
