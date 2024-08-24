from joblib import load

# Load the model
loaded_model = load('random_forest_model.joblib')

import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Start capturing video from the camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # flip the frame
    frame = cv2.flip(frame , 1)
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    # Process the frame
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks: 

        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Drawing the hand landmarks on the captyured frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            handedness_list = []
            
            # Extract handedness
            if results.multi_handedness:
                handedness = results.multi_handedness[hand_index].classification[0].label
            else:
                handedness = None
            encoded_handedness = 1 if handedness == "Right" else 0 if handedness == "Left" else None
            handedness_list.append(encoded_handedness)
            handedness_array = np.array(handedness_list)

                
            # Extract hand landmarks
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            
            landmark_array = np.array(landmarks)
            flattened_land_array = landmark_array.flatten()            

            # Extract hand world landmarks (if available)
            world_landmarks = [(lm.x, lm.y, lm.z) for lm in results.multi_hand_world_landmarks[hand_index].landmark] if results.multi_hand_world_landmarks else []
            
            world_landmark_array = np.array(world_landmarks)
            flattened_world_array = world_landmark_array.flatten()
            
            # print("flatland", flattened_land_array.shape)
            # print("worldland", flattened_world_array.shape)            
            # print("handedness",handedness_array.shape )
            
            row_info = np.concatenate([handedness_array, flattened_land_array, flattened_world_array])
            # print("finalrow", row_info.shape)
            
            row_info_2d = np.array(row_info).reshape(1, -1)

        action_label = loaded_model.predict(row_info_2d)
        # print(action_label, type(action_label))
        # Define a mapping from values to strings
        action_mapping = {
            1: "down",
            0: "front",
            3: "right",
            2: "back",
            4: "left",
            5: "up",
            # Add more mappings as needed
        }

        # Get the value from the array
        value = action_label[0]
        final_label = action_mapping.get(value, None)
        # print(final_label)


        # Display the action label on the frame
        cv2.putText(frame, f'Action:{final_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (205, 0, 0), 2, cv2.LINE_AA)
            

    # Display the frame
    cv2.imshow('Hand Landmarks', (frame))

    # Setup sleep time for frame in live video
    # time.sleep(0.25) 
   
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()