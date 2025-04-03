import cv2
import numpy as np
import mediapipe as mp
import os.path

class FaceDetector:
    def __init__(self, model_selection=1, min_detection_confidence=0.5):
        # Initialize MediaPipe face detection module
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,    # model number 1 seems to work well for settings with multiple faces
            min_detection_confidence=min_detection_confidence
        )


    # Detect the faces from an image
    def detect(self, image_bgr: np.ndarray, padding=0.05) -> list:
        # Convert BGR image to RGB (MediaPipe requires RGB input)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Process the image to detect faces
        results = self.face_detection.process(image_rgb)

        # Extract bounding boxes
        faces = []
        if results.detections:
            for detection in results.detections:
                # Get relative bounding box coordinates
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image_rgb.shape

                # Convert relative coordinates to absolute pixel values
                startX = int(bboxC.xmin * iw)
                startY = int(bboxC.ymin * ih)
                endX = int((bboxC.xmin + bboxC.width) * iw)
                endY = int((bboxC.ymin + bboxC.height) * ih)
                faces.append((startX, startY, endX, endY))

        # Apply the desired padding
        faces = self._get_padded(image_bgr, faces, padding=padding)
        return faces


    # Draw bounding boxes on a copy of the image and return it
    def draw_bounding_boxes(self, image_bgr: np.ndarray, faces: list, padding=0.0) -> np.ndarray:
        image_copy = image_bgr.copy()
        padded_faces = self._get_padded(image_copy, faces, padding=padding)
        for face in padded_faces:
            x1_new, y1_new, x2_new, y2_new = face
            cv2.rectangle(image_copy, (x1_new, y1_new), (x2_new, y2_new), color=(255, 0, 0), thickness=3)
        return image_copy


    # Return the croppings of the faces
    def get_croppings(self, image_bgr: np.ndarray, faces: list, save_croppings=False, save_path=''):
        face_croppings = [image_bgr[face[1]:face[3], face[0]:face[2]] for face in faces]
        if save_croppings:
            os.makedirs(save_path, exist_ok=True)
            for i, face_cropping in enumerate(face_croppings):
                cv2.imwrite(os.path.join(save_path, f'cropping_{i}.png'), face_cropping)
        return face_croppings  # Optional: return croppings if needed


    # Apply padding to a list of rectangles on the image
    def _get_padded(self, image_bgr: np.ndarray, faces: list, padding=0.1):
        faces_padded = []
        for face in faces:
            bbox_width = abs(face[2] - face[0])
            bbox_height = abs(face[3] - face[1])

            x1_new = max(0, int(face[0] - padding * bbox_width))
            x2_new = min(int(face[2] + padding * bbox_width), image_bgr.shape[1])
            y1_new = max(0, int(face[1] - padding * bbox_height))
            y2_new = min(int(face[3] + padding * bbox_height), image_bgr.shape[0])

            faces_padded.append((x1_new, y1_new, x2_new, y2_new))
        return faces_padded


if __name__ == '__main__':
    # Some used directory and image paths
    parent_path = os.path.dirname(os.path.abspath(__file__))
    source_path = os.path.dirname(parent_path)
    base_path = os.path.dirname(source_path)
    test_image_path = os.path.join(base_path, 'tests/data/test_image_0.jpg')
    test_image = cv2.imread(test_image_path)

    detector = FaceDetector()

    # Visually test face detection on a test_image
    faces = detector.detect(test_image, padding=0.5)
    display_image = detector.draw_bounding_boxes(test_image, faces, padding=0.0)  # Draw unpadded boxes
    result_path = os.path.join(base_path, 'results')
    cropping_path = os.path.join(result_path, 'croppings')
    detector.get_croppings(image_bgr=test_image, faces=faces, save_croppings=True, save_path=cropping_path)
    print('Number of people detected:', len(faces))
    cv2.imshow('detected_faces', display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Visually test with live webcam
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break
        faces = detector.detect(frame, padding=0.1)
        display_frame = detector.draw_bounding_boxes(frame, faces, padding=0.0)
        cv2.imshow('Live Face Detection', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()