import unittest
import numpy as np
import cv2
import os
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from facial_recognition.face_detection import FaceDetector


class TestFaceDetector(unittest.TestCase):
    def setUp(self):
        self.detector = FaceDetector()
        self.test_image_0_path = os.path.join(os.path.dirname(__file__), "data", "test_image_0.jpg")
        self.test_image_0 = cv2.imread(self.test_image_0_path)
        self.test_image_1_path = os.path.join(os.path.dirname(__file__), "data", "test_image_1.png")
        self.test_image_1 = cv2.imread(self.test_image_1_path)

    def test_detect_faces_in_image(self):
        # If the test image contains one or more faces
        faces_0 = self.detector.detect(self.test_image_0, padding=0.1)
        faces_1 = self.detector.detect(self.test_image_1, padding=0.1)
        # the first test image has 6 faces
        self.assertGreaterEqual(len(faces_0), 1, "No faces detected in the first test image")
        self.assertEqual(len(faces_0), 6, f"Total number of faces in the first test image is {len(faces_0)} and not 6")
        # the second test image has 1 face
        self.assertLess(len(faces_1), 2, "More than one face detected in the second test image")
        self.assertEqual(len(faces_1), 1, f"Total number of faces in the secnod test image is {len(faces_1)} and not 1")


    def test_draw_bounding_boxes(self):
        faces = self.detector.detect(self.test_image_0)
        image_with_boxes = self.detector.draw_bounding_boxes(self.test_image_0, faces)
        # Check that the output image has the same shape as input
        self.assertEqual(image_with_boxes.shape, self.test_image_0.shape)
        # And that drawing changed the image (at least in some pixels)
        self.assertFalse(np.array_equal(self.test_image_0, image_with_boxes))

    def test_get_croppings(self):
        faces = self.detector.detect(self.test_image_0)
        croppings = self.detector.get_croppings(self.test_image_0, faces)
        # Ensure that we get a cropping for each face detected
        self.assertEqual(len(croppings), len(faces))
        # Verify the dimensions of the first cropping
        if len(croppings) > 0:
            x1, y1, x2, y2 = faces[0]
            expected_height = y2 - y1
            expected_width = x2 - x1
            self.assertEqual(croppings[0].shape[0], expected_height)
            self.assertEqual(croppings[0].shape[1], expected_width)

if __name__ == '__main__':
    unittest.main()
