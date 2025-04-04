import unittest
import numpy as np
import cv2
import os
from unittest.mock import patch, MagicMock
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from database import profile_creator

class TestProfileCreator(unittest.TestCase):
    @patch('database.profile_creator.take_picture')
    @patch('database.profile_creator.cv2.imshow')
    @patch('database.profile_creator.cv2.waitKey')
    @patch('database.profile_creator.cv2.destroyAllWindows')
    @patch('database.database_manager.ImageDatabase.add_face')
    @patch('builtins.input', side_effect=["Charlie", "003"])
    def test_create_profile(self, mock_input, mock_add_face, mock_destroy, mock_waitKey, mock_imshow, mock_take_picture):
        # Use a dummy image for testing profile creation
        dummy_image_path = os.path.join(os.path.dirname(__file__), 'data', 'test_image_1.png')
        dummy_image = cv2.imread(dummy_image_path)
        mock_take_picture.return_value = dummy_image
        # Run create_profile which should use the dummy image and provided input
        profile_creator.create_profile(image=dummy_image, collection="test_faces")
        self.assertTrue(mock_add_face.called, "Expected add_face to be called in create_profile")


if __name__ == '__main__':
    unittest.main()
