import unittest
import numpy as np
import cv2
import os
from unittest.mock import patch
import mediapipe as mp
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from engagement_monitoring.perclos import PerclosCalculator

# Dummy classes to simulate MediaPipe landmarks
class DummyLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class DummyLandmarks:
    def __init__(self):
        self.landmark = [DummyLandmark(0.3, 0.3) for _ in range(468)]

class TestPerclosCalculator(unittest.TestCase):
    def setUp(self):
        # Use a smaller window for testing purposes
        self.calculator = PerclosCalculator(perclos_window=10, ear_threshold=0.2, max_distance=100, max_missed_frames=5)
        # Load a test image containing a face
        self.test_image_path = os.path.join(os.path.dirname(__file__), "data", "test_image_1.png")
        self.test_frame = cv2.imread(self.test_image_path)

    def test_calculate_EAR_returns_float(self):
        dummy_landmarks = DummyLandmarks()
        # Call calculate_EAR and verify it returns a float value
        ear = self.calculator.calculate_EAR(dummy_landmarks, crop_width=100, crop_height=100)
        self.assertIsInstance(ear, float)

    @patch('engagement_monitoring.perclos.FaceDetector.detect', return_value=[(10, 10, 110, 110)])
    @patch.object(PerclosCalculator, '_update_eye_state')
    def test_process_frame(self, mock_update_eye_state, mock_detect):
        perclos_scores = self.calculator.process_frame(self.test_frame)
        # The process_frame method should return a list (possibly empty) of face dictionaries
        self.assertIsInstance(perclos_scores, list)
        # If any faces are tracked, they should have a 'bbox' key
        if perclos_scores:
            self.assertIn('bbox', perclos_scores[0])

    def test_calculate_EAR(self):
        """Test that EAR for open eyes is greater than for closed eyes using real images."""
        face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

        # Load test images
        open_eyes_img = cv2.imread(os.path.join(os.path.dirname(__file__), "data", "test_open_eyes.jpg"))
        closed_eyes_img = cv2.imread(os.path.join(os.path.dirname(__file__), "data", "test_closed_eyes.jpg"))

        def get_landmarks(image):
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)
            return results.multi_face_landmarks[0] if results.multi_face_landmarks else None

        # Extract landmarks
        landmarks_open = get_landmarks(open_eyes_img)
        landmarks_closed = get_landmarks(closed_eyes_img)

        # Ensure that landmarks were detected
        self.assertIsNotNone(landmarks_open, "No landmarks detected in open eyes image")
        self.assertIsNotNone(landmarks_closed, "No landmarks detected in closed eyes image")

        # Calculate EAR
        ear_open = self.calculator.calculate_EAR(landmarks_open, open_eyes_img.shape[1], open_eyes_img.shape[0])
        ear_closed = self.calculator.calculate_EAR(landmarks_closed, closed_eyes_img.shape[1], closed_eyes_img.shape[0])

        # Ensure EAR values are valid
        self.assertFalse(np.isnan(ear_open), "EAR for open eyes is NaN")
        self.assertFalse(np.isnan(ear_closed), "EAR for closed eyes is NaN")

        # Assert that EAR for open eyes is greater than closed eyes
        self.assertGreater(ear_open, ear_closed, "EAR for open eyes should be greater than for closed eyes")

if __name__ == '__main__':
    unittest.main()
