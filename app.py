import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

# 1. Load your saved model
model = load_model('C:\\Users\\Sir\\Desktop\\FINAL TRY\\emotion_detection_model.h5')

# 2. Define emotion labels in the same order used during training
emotions = [
    'Angry',    # index 0
    'Disgust',  # index 1
    'Fear',     # index 2
    'Happy',    # index 3
    'Sad',      # index 4
    'Surprise', # index 5
    'Neutral'   # index 6
]

# 3. Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('C:\\Users\\Sir\\Desktop\\FINAL TRY\\haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam")
    exit()

print("Press 'c' to capture an image and analyze.")
print("Press 'q' to quit.")

while True:
    # Read frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Show the live feed
    cv2.imshow('Live Feed', frame)

    # Listen for key presses
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        # --- Capture & analyze the current frame ---
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        # Copy the frame so we can draw on it
        analyzed_frame = frame.copy()
        
        for (x, y, w, h) in faces:
            # Crop the face
            face_img = gray[y:y+h, x:x+w]
            
            # Resize to 48x48 (if your model uses 48x48)
            face_img = cv2.resize(face_img, (48, 48))
            
            # Normalize
            face_img = face_img / 255.0
            
            # Reshape for model input: (1, 48, 48, 1)
            face_img = np.reshape(face_img, (1, 48, 48, 1))
            
            # Predict emotion
            predictions = model.predict(face_img)
            predicted_class = np.argmax(predictions)
            predicted_emotion = emotions[predicted_class]
            
            # Draw a rectangle around the face
            cv2.rectangle(analyzed_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Put text (emotion label)
            cv2.putText(analyzed_frame,
                        predicted_emotion,
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 0, 0),
                        2)
        
        # Show the captured & analyzed image in a new window
        cv2.imshow('Captured & Analyzed', analyzed_frame)

    elif key == ord('q'):
        # Quit the loop
        break

# Release resources
cap.release()
cv2.destroyAllWindows()