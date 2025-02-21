import cv2
import mediapipe as mp
import math
import numpy as np
import sounddevice as sd
import time

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# OpenCV video capture
cap = cv2.VideoCapture(1)

# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Function to generate smooth sine wave sound
def generate_sine_wave(frequency, duration, volume, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = np.sin(2 * np.pi * frequency * t)
    return waveform * volume

# Function to play rhythmic sound (bass or melody)
def play_sound(frequency, volume, duration, rhythm=0.25, sample_rate=44100):
    samples = generate_sine_wave(frequency, duration, volume, sample_rate)
    rhythm_samples = int(rhythm * sample_rate)  # Rhythm interval (smooth timing)
    for i in range(0, len(samples), rhythm_samples):
        sd.play(samples[i:i+rhythm_samples], samplerate=sample_rate)
        sd.wait()

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

        # Default frequency and volume
        bass_frequency = 110  # Bass (Left hand)
        melody_frequency = 440  # Melody (Right hand)
        volume = 0.1  # Default volume

        if results.multi_hand_landmarks:
            for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                h, w, _ = image.shape
                landmarks = {}

                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks[id] = (cx, cy)

                # Identify left and right hands
                if 4 in landmarks and 8 in landmarks:
                    thumb = landmarks[4]
                    index_finger = landmarks[8]

                    # Calculate distance between thumb and index
                    distance = calculate_distance(thumb, index_finger)

                    # Hand position for pitch control
                    if hand_index == 0:  # Left Hand (Bassline rhythm)
                        bass_frequency = 110 + (distance * 10)  # Adjust frequency based on thumb/index distance
                    else:  # Right Hand (Melody)
                        melody_frequency = 440 + (distance * 5)  # Adjust frequency for melody

                    # Draw line between thumb and index
                    cv2.line(image, thumb, index_finger, (0, 255, 0), 2)

                # Use hand height for volume control
                if 9 in landmarks:
                    hand_height = landmarks[9][1]  # Vertical position of middle finger
                    volume = max(0.01, min(1, (hand_height / h)))  # Volume from 0.01 to 1

                # Play sound based on hand movement
                if hand_index == 0:  # Left Hand (Bassline)
                    play_sound(bass_frequency, volume, 0.25, rhythm=0.25)
                else:  # Right Hand (Melody)
                    play_sound(melody_frequency, volume, 0.25, rhythm=0.25)

                # Draw landmarks and connections
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the result
        cv2.imshow('MediaPipe Hands - Rhythmic Sound Control', image)

        if cv2.waitKey(5) & 0xFF in [27, ord('q')]:  # Exit on ESC or 'q'
            break

cap.release()
cv2.destroyAllWindows()
