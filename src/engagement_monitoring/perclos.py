import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from facial_recognition.face_detection import FaceDetector

class PerclosCalculator:
    def __init__(self, perclos_window=100, ear_threshold=0.2, max_distance=50, max_missed_frames=5):
        '''
        Arguments:
        - perclos_window (int):    Window size for PERCLOS calculation (number of frames).
        - ear_threshold (float):   Threshold for EAR below which eyes are considered closed.
        - max_distance (float):    Maximum distance (pixels) to match faces across frames.
        - max_missed_frames (int): Number of frames a face can be undetected before removal.
        '''
        # Load our custom FaceDetector module
        self.face_detector = FaceDetector()

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=20,       # Can adjust based on expected number of faces
            refine_landmarks=True,  # For more accurate eye landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.perclos_window = perclos_window
        self.ear_threshold = ear_threshold
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.tracked_faces = []  # List of tracked faces
        self.next_id = 0         # For assigning unique IDs to new faces


    # Calculate the average eye aspect ratio (EAR) value
    def calculate_EAR(self, landmarks, crop_width, crop_height):
        # Define landmark indices for left and right eyes
        left_eye_indices = [362, 385, 387, 263, 373, 380]  # p1, p2, p3, p4, p5, p6
        right_eye_indices = [33, 160, 158, 133, 153, 144]

        # Helper func to convert normalized landmark coordinates to pixel coordinates.
        def get_point(index):
            lm = landmarks.landmark[index]
            return (lm.x * crop_width, lm.y * crop_height)

        # Extract points for left eye and right eye
        p1_left, p2_left, p3_left, p4_left, p5_left, p6_left = [get_point(coord) for coord in left_eye_indices]
        p1_right, p2_right, p3_right, p4_right, p5_right, p6_right = [get_point(coord) for coord in right_eye_indices]

        # Calculate EAR ratio for left and right eye
        ear_left = (np.linalg.norm(np.array(p2_left) - np.array(p6_left)) + np.linalg.norm(np.array(p3_left) - np.array(p5_left))) / (2 * np.linalg.norm(np.array(p1_left) - np.array(p4_left)))
        ear_right = (np.linalg.norm(np.array(p2_right) - np.array(p6_right)) + np.linalg.norm(np.array(p3_right) - np.array(p5_right))) / (2 * np.linalg.norm(np.array(p1_right) - np.array(p4_right)))

        # Return average EAR
        return (ear_left + ear_right) / 2


    # Update the eye state based on EAR
    def _update_eye_state(self, track_face, frame):
        bbox = track_face['bbox']
        # Crop the face from the frame
        crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if crop.size == 0:  # Check if crop is valid
            return

        # Convert to RGB for MediaPipe
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(crop_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0] # Process only the first face in the crop
            # Calculate EAR
            ear = self.calculate_EAR(face_landmarks, crop.shape[1], crop.shape[0])
            eyes_closed = ear < self.ear_threshold
            track_face['eye_states'].append(eyes_closed)
            track_face['ear'] = ear  # Store the latest EAR value
            track_face['landmarks'] = face_landmarks  # Store landmarks for visualization


    # Processes video frame to calculate the PERCLOS score and return a dictionary with relevant detected data
    def process_frame(self, frame):
        # Detect faces
        faces = self.face_detector.detect(frame)
        detected_centers = [((face[0] + face[2]) / 2, (face[1] + face[3]) / 2) for face in faces]

        # Match detected faces to tracked faces
        assigned_tracked_ids = set()
        matched_pairs = []

        for det_idx, det_center in enumerate(detected_centers):
            min_dist = float('inf')
            match_idx = -1
            for track_idx, track_face in enumerate(self.tracked_faces):
                if track_face['id'] in assigned_tracked_ids:
                    continue
                track_center = track_face['center']
                dist = np.sqrt((det_center[0] - track_center[0])**2 +
                               (det_center[1] - track_center[1])**2)
                if dist < min_dist and dist < self.max_distance:
                    min_dist = dist
                    match_idx = track_idx
            if match_idx != -1:
                matched_pairs.append((det_idx, match_idx))
                assigned_tracked_ids.add(self.tracked_faces[match_idx]['id'])

        # Update matched tracked faces
        for det_idx, track_idx in matched_pairs:
            self.tracked_faces[track_idx]['bbox'] = faces[det_idx]
            self.tracked_faces[track_idx]['center'] = detected_centers[det_idx]
            self.tracked_faces[track_idx]['missed_frames'] = 0
            self._update_eye_state(self.tracked_faces[track_idx], frame)

        # Add new detected faces
        assigned_det_indices = {det_idx for det_idx, _ in matched_pairs}
        for det_idx in range(len(faces)):
            if det_idx not in assigned_det_indices:
                new_face = {
                    'id': self.next_id,
                    'bbox': faces[det_idx],
                    'center': detected_centers[det_idx],
                    'eye_states': deque(maxlen=self.perclos_window),
                    'missed_frames': 0
                }
                self.next_id += 1
                self.tracked_faces.append(new_face)
                self._update_eye_state(new_face, frame)

        # Update missed frames for unmatched tracked faces
        for track_face in self.tracked_faces[:]:  # Copy to allow modification
            if track_face['id'] not in assigned_tracked_ids:
                track_face['missed_frames'] += 1

        # Remove faces with too many missed frames
        self.tracked_faces = [f for f in self.tracked_faces if f['missed_frames'] < self.max_missed_frames]

        # Calculate PERCLOS scores
        perclos_scores = []
        for track_face in self.tracked_faces:
            if len(track_face['eye_states']) > 0:
                perclos = (sum(track_face['eye_states']) / len(track_face['eye_states'])) * 100
            else:
                perclos = 0.0
            perclos_scores.append({
                'bbox': track_face['bbox'],
                'perclos': perclos,
                'ear': track_face.get('ear', 0.0),  # Latest EAR value
                'landmarks': track_face.get('landmarks', None)  # Landmarks for visualization
            })
        return perclos_scores
    

# To draw the eye boundary landmarks
def draw_landmarks(frame, landmarks, bbox):
    # return if no landmarks
    if landmarks is None:
        return

    # Define landmark indices for left and right eyes (based on MediaPipe FaceMesh standards)
    left_eye_indices = [362, 385, 387, 263, 373, 380]
    right_eye_indices = [33, 160, 158, 133, 153, 144]

    # Helper function to convert normalized coordinates to frame coordinates
    def get_point(index):
        lm = landmarks.landmark[index]
        # Scale to bounding box size and offset by bbox position
        x = int((lm.x * (bbox[2] - bbox[0])) + bbox[0])
        y = int((lm.y * (bbox[3] - bbox[1])) + bbox[1])
        return (x, y)

    # Draw points for left and right eyes
    for idx in [*left_eye_indices, *right_eye_indices]:
        point = get_point(idx)
        cv2.circle(frame, point, 2, (0, 255, 0), -1)  # Green dots


# Visually testing it on webcam input
if __name__ == '__main__':

    # Initialize the calculator
    calculator = PerclosCalculator(
        perclos_window=100,           # Window size of 100 frames
        ear_threshold=0.2,  # EAR threshold for eye closure
        max_distance=50,    # Max distance for face matching
        max_missed_frames=5  # Frames before removing a face
    )

    # Test on webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    show_bbox = True  # Set to True to display bounding boxes and PERCLOS scores

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Process the frame to get PERCLOS, EAR, and landmarks
        perclos_scores = calculator.process_frame(frame)

        if show_bbox:
            for item in perclos_scores:
                bbox = item['bbox']
                perclos = item['perclos']
                ear = item['ear']
                landmarks = item['landmarks']

                # Draw bounding box and eye landmarks
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                draw_landmarks(frame, landmarks, bbox)

                # Display PERCLOS and EAR scores above the bounding box
                text = f"PERCLOS: {perclos:.2f}% | EAR: {ear:.2f}"
                cv2.putText(frame, text, (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('PERCLOS Calculation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()