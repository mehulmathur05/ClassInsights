import cv2
import numpy as np
import mediapipe as mp


class FaceDetector:
    def __init__(self, model_selection=1, min_detection_confidence=0.5):

        # Initialize MediaPipe face detection module
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence
        )


    # Detect the faces from an image
    def detect(self, image_bgr: np.ndarray, show_bbox=False) -> list:

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
                ih, iw, _ = image_rgb.shape  # Image height and width
                # Convert relative coordinates to absolute pixel values
                startX = int(bboxC.xmin * iw)
                startY = int(bboxC.ymin * ih)
                endX = int((bboxC.xmin + bboxC.width) * iw)
                endY = int((bboxC.ymin + bboxC.height) * ih)
                faces.append((startX, startY, endX, endY))

        if show_bbox: self._display_bbox(image_bgr, faces, padding=0.1)

        return faces


    def _display_bbox(self, image_bgr: np.ndarray, faces : list, padding = 0.1):
        for face in faces:
            bbox_width = abs(face[2] - face[0])
            bbox_height = abs(face[3] - face[1])

            x1_new = max(0, int(face[0] - padding*bbox_width))
            x2_new = min(int(face[2] + padding*bbox_width), image_bgr.shape[1])
            y1_new = max(0, int(face[1] - padding*bbox_height))
            y2_new = min(int(face[3] + padding*bbox_height), image_bgr.shape[0])
            
            cv2.rectangle(image_bgr, (x1_new, y1_new), (x2_new, y2_new), color=(255, 0, 0), thickness=3)


if __name__ == '__main__':

    test_image_path = 'tests/data/test_image_0.jpg'
    test_image = cv2.imread(test_image_path)

    detector = FaceDetector()
    faces = detector.detect(test_image, show_bbox=True)

    print('Number of people detected:', len(faces))
    cv2.imshow('detected_faces', test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
