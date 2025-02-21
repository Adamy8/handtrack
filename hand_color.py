import cv2
import mediapipe as mp
import math
import numpy as np

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# OpenCV video capture
cap = cv2.VideoCapture(1)

# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Setup MediaPipe Hands
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image and convert color
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image with MediaPipe Hands
        results = hands.process(image)

        # Convert the image back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Default gradient values
        red_green_intensity = 50
        blue_purple_intensity = 50

        if results.multi_hand_landmarks:
            for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                h, w, _ = image.shape

                # Dictionary to store landmark positions
                landmarks = {}

                # Iterate through hand landmarks
                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks[id] = (cx, cy)

                    # Draw text with ID and confidence
                    confidence = lm.z  # Depth info (not visibility in hands)
                    cv2.putText(image, f'{id}({confidence:.2f})', (cx, cy), 
                                cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

                # Identify left and right hands
                if 4 in landmarks and 8 in landmarks:
                    thumb = landmarks[4]
                    index_finger = landmarks[8]

                    # Calculate distance
                    distance = calculate_distance(thumb, index_finger)

                    # Assign different gradient effects to each hand
                    if hand_index == 0:  # First detected hand (assumed left)
                        red_green_intensity = min(255, int(distance * 2))
                    else:  # Second detected hand (assumed right)
                        blue_purple_intensity = min(255, int(distance * 2))

                    # Draw line between thumb and index finger
                    cv2.line(image, thumb, index_finger, (0, 255, 0), 2)

                    # Display distance value
                    cv2.putText(image, f'Distance: {distance:.2f}', (10, 30 + hand_index * 30), 
                                cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

                # Draw landmarks and connections
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # **Apply Smooth Gradient Effect**
        gradient_overlay = np.zeros_like(image, dtype=np.uint8)
        height, width, _ = image.shape

        # Create the left-hand (red-green) gradient
        for y in range(height):
            red = int((y / height) * red_green_intensity)
            green = int((1 - y / height) * red_green_intensity)
            gradient_overlay[y, :, :] = [0, green, red]  # OpenCV uses BGR

        # Create the right-hand (blue-purple) gradient
        for x in range(width):
            blue = int((x / width) * blue_purple_intensity)
            purple = int((1 - x / width) * blue_purple_intensity)

            # FIXED: Convert before adding to avoid type mismatch error
            gradient_overlay[:, x, :] = gradient_overlay[:, x, :].astype(np.uint16) + np.array([blue, 0, purple], dtype=np.uint16)
        
        # Ensure values stay within [0, 255] range and convert back to uint8
        gradient_overlay = np.clip(gradient_overlay, 0, 255).astype(np.uint8)

        # Blend the gradient with the image
        cv2.addWeighted(gradient_overlay, 0.5, image, 0.5, 0, image)

        # Show the result
        cv2.imshow('MediaPipe Hands - Dual Gradient Control', image)

        if cv2.waitKey(5) & 0xFF in [27, ord('q')]:  # Exit on ESC or 'q'
            break

cap.release()
cv2.destroyAllWindows()
