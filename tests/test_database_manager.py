import unittest
import numpy as np
import cv2
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from database.database_manager import ImageDatabase

class TestImageDatabase(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory to simulate persistence
        self.temp_dir = tempfile.mkdtemp()
        self.db = ImageDatabase(persistence_path=self.temp_dir)
        # Patch face_model.get to simulate face embedding extraction
        self.fake_face = MagicMock()
        self.fake_face.embedding = np.array([0.1, 0.2, 0.3])
        self.get_patch = patch.object(self.db.face_model, 'get', return_value=[self.fake_face])
        self.mock_get = self.get_patch.start()

    def tearDown(self):
        self.get_patch.stop()
        shutil.rmtree(self.temp_dir)

    def test_add_face_single_face(self):
        # Use an image from tests/data (for example, a cropped face image)
        face_img_path = os.path.join(os.path.dirname(__file__), "data", "test_image_1.png")
        face_img = cv2.imread(face_img_path)
        # Should work without raising an error if one face is detected
        try:
            self.db.add_face(face_img, roll_number="001", name="A")
        except Exception as e:
            self.fail(f"add_face raised an exception unexpectedly: {e}")

    def test_add_face_multiple_faces(self):
        # Simulate a scenario where multiple faces are returned by making a list of duplicate faces
        with patch.object(self.db.face_model, 'get', return_value=[self.fake_face, self.fake_face]):
            face_img_path = os.path.join(os.path.dirname(__file__), "data", "test_image_1.png")
            face_img = cv2.imread(face_img_path)
            with self.assertRaises(ValueError):
                self.db.add_face(face_img, roll_number="002", name="B")

    def test_query_face_no_match(self):
        # Simulate no face being detected during query by returning an empty list
        with patch.object(self.db.face_model, 'get', return_value=[]):
            face_img_path = os.path.join(os.path.dirname(__file__), "data", "test_image_1.png")
            face_img = cv2.imread(face_img_path)
            result = self.db.query_face(face_img)
            self.assertEqual(result, (None, None))


if __name__ == '__main__':
    unittest.main()
