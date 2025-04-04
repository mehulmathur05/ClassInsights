import cv2
import pandas as pd
import numpy as np
from datetime import datetime, time
import os
from facial_recognition.face_detection import FaceDetector
from database.database_manager import ImageDatabase
from database.profile_creator import create_profile
from engagement_monitoring.perclos import PerclosCalculator, draw_landmarks

class Pipeline:
    def __init__(self, schedule_path='../data/schedule.csv', results_dir='../results', k=30, collection="faces"):
        '''
        Arguments:
        - schedule_path (str): Path to the schedule file
        - results_dir (str): Directory for storing reports and attendance files
        - k (int): Number of frames after which to update the present list
        - collection (str): Name of the chromaDB collection to create or use
        '''
        self.face_detector = FaceDetector()
        self.db = ImageDatabase(collection=collection)
        self.collection = collection
        self.perclos_calculator = PerclosCalculator()
        self.schedule_path = schedule_path
        self.results_dir = results_dir
        self.k = k
        os.makedirs(results_dir, exist_ok=True)
        self._reset_files()
        self.schedule = pd.read_csv(schedule_path)
        print("Schedule DataFrame sample:\n", self.schedule.head())
        self.all_students = self._get_all_students()
        self.processed_classes = set()  # Track classes processed in the current run


    # Remove the pre-existing report and attendance files
    def _reset_files(self):
        report_path = os.path.join(self.results_dir, 'report.csv')
        attendance_path = os.path.join(self.results_dir, f'attendance_{datetime.now().date().strftime("%Y-%m-%d")}.csv')
        if os.path.exists(report_path): os.remove(report_path)
        if os.path.exists(attendance_path): os.remove(attendance_path)


    # Get the name and roll number of all students in the database
    def _get_all_students(self):
        collections = self.db.collection.get()
        return {item['roll_number']: item['name'] for item in collections['metadatas']}


    # Determine current class using time data from the schedule.csv file
    def _get_current_class(self):
        now = datetime.now().time()
        today = datetime.now().date()
        for _, row in self.schedule.iterrows():
            start_time = datetime.strptime(row['start_time'], '%H:%M:%S').time()
            end_time = datetime.strptime(row['end_time'], '%H:%M:%S').time()
            if start_time <= now <= end_time:
                return {
                    'subject_name': row['subject_name'],
                    'subject_code': row['subject_code'],
                    'end_time': end_time,
                    'date': today
                }
        return None


    # Update or create the attendance file
    def _update_attendance_csv(self, class_info, detected_rolls, avg_perclos_scores):
        date_str = class_info['date'].strftime('%Y-%m-%d')
        filename = os.path.join(self.results_dir, f'attendance_{date_str}.csv')
        
        if os.path.exists(filename):
            df = pd.read_csv(filename)
        else:
            df = pd.DataFrame({'roll_number': list(self.all_students.keys()), 
                               'name': list(self.all_students.values())})
        
        subject_code = class_info['subject_code']
        presence_col = f'{subject_code}_present'
        perclos_col = f'{subject_code}_interactiveness'
        
        if presence_col not in df.columns:
            df[presence_col] = False
        if perclos_col not in df.columns:
            df[perclos_col] = 0.0
        
        for roll, perclos in zip(detected_rolls, avg_perclos_scores):
            if roll in df['roll_number'].values:
                df.loc[df['roll_number'] == roll, presence_col] = True
                df.loc[df['roll_number'] == roll, perclos_col] = 100 - perclos
        
        df.to_csv(filename, index=False)


    # Append to report.csv sequentially without modifying existing rows
    def _generate_report_csv(self, class_info, detected_rolls, avg_perclos_scores):
        file_path = os.path.join(self.results_dir, 'report.csv')
        class_key = f"{class_info['subject_code']}_{class_info['date'].strftime('%Y-%m-%d')}"
        
        if class_key not in self.processed_classes:
            report = {
                'subject_name': class_info['subject_name'],
                'subject_code': class_info['subject_code'],
                'date': class_info['date'].strftime('%Y-%m-%d'),
                'attendance': len(detected_rolls),
                'interactiveness': np.mean([100 - p for p in avg_perclos_scores]) if avg_perclos_scores else 0.0
            }
            
            # Append mode ensures sequential addition
            df = pd.DataFrame([report])
            df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
            self.processed_classes.add(class_key)


    # Function to run the end to end pipeline
    def run(self):
        while True:
            response = input("Would you like to add new students to the database? (yes/no): ").strip().lower()
            if response == 'yes':
                create_profile(collection=self.collection)
            elif response == 'no':
                break
            else:
                print("Please enter 'yes' or 'no'.")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        current_class = None
        detected_rolls = set()
        perclos_history = {}
        temp_detected_rolls = set()
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break

            frame_count += 1

            new_class = self._get_current_class()
            
            # Force class end if current time exceeds end_time
            if current_class is not None:
                now = datetime.now().time()
                if now > current_class['end_time']:
                    new_class = None

            if new_class != current_class:
                if current_class is not None:  # Class ended
                    detected_rolls.update(temp_detected_rolls)
                    temp_detected_rolls.clear()
                    if detected_rolls:
                        avg_perclos_scores = [np.mean(perclos_history.get(roll, [0])) for roll in detected_rolls]
                        self._generate_report_csv(current_class, detected_rolls, avg_perclos_scores)
                        self._update_attendance_csv(current_class, detected_rolls, avg_perclos_scores)
                    else:
                        print("No students detected for this class.")
                    detected_rolls.clear()
                    perclos_history.clear()
                    frame_count = 0
                    print(f"Class {current_class['subject_code']} ended. Reports updated. PERCLOS reset.")
                current_class = new_class

            if current_class is None:
                cv2.putText(frame, "No class scheduled", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Class Monitoring', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Use PerclosCalculator for both detection and scoring
            perclos_scores = self.perclos_calculator.process_frame(frame)

            for score_item in perclos_scores:
                bbox = score_item['bbox']  
                bbox_padded = self.face_detector._get_padded(frame, [score_item['bbox']], padding=0.4)[0] # [x_min, y_min, x_max, y_max]
                cropping_padded = frame[bbox_padded[1]:bbox_padded[3], bbox_padded[0]:bbox_padded[2]]
                roll, name = self.db.query_face(cropping_padded)
                if roll:
                    temp_detected_rolls.add(roll)
                    perclos_history.setdefault(roll, []).append(score_item['perclos'])
                    current_avg_perclos = np.mean(perclos_history[roll])
                    interactiveness = 100 - current_avg_perclos
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                    draw_landmarks(frame, score_item['landmarks'], bbox)
                    text = f"{name} | Interactiveness: {interactiveness:.2f}%"
                    cv2.putText(frame, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if frame_count % self.k == 0 or frame_count == 1:
                detected_rolls.update(temp_detected_rolls)
                temp_detected_rolls.clear()

            class_text = f"Class: {current_class['subject_name']} ({current_class['subject_code']}) | Attendance: {len(detected_rolls)}"
            cv2.putText(frame, class_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Class Monitoring', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if current_class is not None:
            detected_rolls.update(temp_detected_rolls)
            if detected_rolls:
                avg_perclos_scores = [np.mean(perclos_history.get(roll, [0])) for roll in detected_rolls]
                self._generate_report_csv(current_class, detected_rolls, avg_perclos_scores)
                self._update_attendance_csv(current_class, detected_rolls, avg_perclos_scores)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # Load and run the pipeline
    pipeline = Pipeline(k=30, collection="faces0")
    pipeline.run()