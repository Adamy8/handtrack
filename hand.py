import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# OpenCV video capture
cap = cv2.VideoCapture(1)

# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Function to calculate the angle between the line connecting two points and the vertical
def calculate_angle(point1, point2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    return math.degrees(math.atan2(dy, dx))

# Setup MediaPipe Hands
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
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

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
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

                # Draw a line between the thumb and all four fingers
                if 4 in landmarks:
                    thumb = landmarks[4]
                    for finger_id in [8, 12, 16, 20]:  # Index, Middle, Ring, Pinky
                        if finger_id in landmarks:
                            cv2.line(image, thumb, landmarks[finger_id], (0, 255, 0), 2)

                # Draw a line between thumb tip and index tip
                if 4 in landmarks and 8 in landmarks:
                    cv2.line(image, landmarks[4], landmarks[8], (0, 0, 255), 2)
                    distance = calculate_distance(landmarks[4], landmarks[8])
                    angle = calculate_angle(landmarks[4], landmarks[8])

                    # Display distance and angle on the top left corner
                    cv2.putText(image, f'Distance: {distance:.2f}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
                    cv2.putText(image, f'Angle: {angle:.2f}', (10, 60), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

                # Draw landmarks and connections
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the result
        cv2.imshow('MediaPipe Hands - Thumb to Fingers Lines', image)

        if cv2.waitKey(5) & 0xFF in [27, ord('q')]:  # Exit on ESC or 'q'
            break

cap.release()
cv2.destroyAllWindows()
