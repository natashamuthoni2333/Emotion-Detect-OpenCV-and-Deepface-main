import cv2
from deepface import DeepFace
import os

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Directory containing images
image_directory = 'E:\COMPUTER VISION\PROJECTS CV\Emotion-Detect-OpenCV-and-Deepface-main\Emotion-Detect-OpenCV-and-Deepface-main\Test_Images'

# Iterate through all images in the directory
for filename in os.listdir(image_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust the file extensions as needed
        image_path = r'E:\COMPUTER VISION\PROJECTS CV\Emotion-Detect-OpenCV-and-Deepface-main\Emotion-Detect-OpenCV-and-Deepface-main\Test_Images\e2.jpeg'
       
        # Load image
        image = cv2.imread(image_path)

        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Convert grayscale image to RGB format
        rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = rgb_image[y:y + h, x:x + w]

            # Perform emotion analysis on the face ROI
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            # Determine the dominant emotion
            emotion = result[0]['dominant_emotion']

            # Draw rectangle around face and label with predicted emotion
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the resulting image
        cv2.imshow('Emotion Detection', image)
       
        # Wait for a key press and close the image window
        cv2.waitKey(0)
        cv2.destroyAllWindows()