import unittest
import os
import tempfile
import shutil
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock

import sys

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from pipeline import Pipeline

class TestPipeline(unittest.TestCase):
    def setUp(self):
        # Create temporary directories and a fake schedule CSV
        self.temp_results = tempfile.mkdtemp()
        self.temp_schedule = os.path.join(self.temp_results, "schedule.csv")
        with open(self.temp_schedule, "w") as f:
            f.write("subject_name,subject_code,start_time,end_time\n")
            f.write("Physics,PH101,00:00:00,23:59:59\n")
        self.pipeline = Pipeline(schedule_path=self.temp_schedule, results_dir=self.temp_results, k=5, collection="test_faces")
        # Patch the database collection.get to return dummy student data
        dummy_collection = {'metadatas': [{'roll_number': '004', 'name': 'Dana'}]}
        self.pipeline.db.collection.get = MagicMock(return_value=dummy_collection)
        self.pipeline.all_students = self.pipeline._get_all_students()

    def tearDown(self):
        shutil.rmtree(self.temp_results)

    def test_get_all_students(self):
        expected = {'004': 'Dana'}
        self.assertEqual(self.pipeline.all_students, expected)

    def test_reset_files(self):
        self.pipeline._reset_files()
        report_file = os.path.join(self.temp_results, 'report.csv')
        date = datetime.now().date().strftime('%Y-%m-%d')
        attendance_file = os.path.join(self.temp_results, f'attendance_{date}.csv')
        self.assertTrue(os.path.exists(report_file))
        self.assertTrue(os.path.exists(attendance_file))

    @patch('pipeline.cv2.VideoCapture')
    @patch('pipeline.input', side_effect=["no"])
    def test_run_pipeline_no_webcam(self, mock_input, mock_VideoCapture):
        # Simulate failure to open webcam
        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = False
        mock_VideoCapture.return_value = cap_mock
        # run should return early if webcam fails
        self.pipeline.run()

if __name__ == '__main__':
    unittest.main()
