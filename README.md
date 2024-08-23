# Hand Gesture Recognition

This project implements a machine learning model to recognize six specific hand gestures in real-time using a live video feed.

## Objective
The primary objective of this project is to develop a machine learning model that can recognize six specific hand gestures in real-time using a live video feed. The gestures include various common actions, and the goal is to achieve high accuracy and real-time performance to ensure practical usability.

## Methodology

### Data Collection and Preprocessing
#### Data Collection
- Captured multiple photos of hand gestures representing six different actions
- Used a standard camera to take images of a single hand performing each gesture
- Recorded gestures from various angles and under different lighting conditions
- Collected at least 100 images per gesture

#### Data Augmentation
- Performed augmentation techniques to enhance dataset diversity:
  - Rotation
  - Scaling
  - Brightness and Contrast Adjustment

#### CSV File
- Organized dataset by saving file paths and gesture labels in a CSV file

### Feature Extraction
- Used Mediapipe's hand solutions for hand tracking and landmark detection
- Extracted 21 key points (landmarks) on the hand
- Created feature arrays including landmarks and handedness

### Model Training
- Used RandomForestClassifier for gesture recognition
- Split dataset into 80% training and 20% testing
- Optimized hyperparameters for best performance

### Model Deployment
- Saved trained model using joblib library
- Implemented real-time prediction system using cv2 for video capture

## Future Improvements
- Expand the dataset with more diverse hand gestures
- Experiment with deep learning models for potentially higher accuracy
- Implement the system on mobile devices for wider accessibility

## Contributors
-Aryav Gupta(https://github.com/AryavGupta)
