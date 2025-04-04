import cv2
from database.database_manager import ImageDatabase
from facial_recognition.face_detection import FaceDetector

# Create a profile 
def create_profile(image=None, collection="faces"):
    # Input the name and roll num
    name = input("Enter student name: ")
    roll_num = input("Enter student roll number: ")

    # Initialize the database and face detector instance
    db = ImageDatabase(collection=collection)
    detector = FaceDetector()

    # Take new picture if image not provided
    if image is None:
        image = take_picture()

    faces = detector.detect(image, padding=0.4)

    # Generate the face cropping
    cropping = detector.get_croppings(image, faces)[0]
    cv2.imshow('captured image', cropping)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cropping = cv2.cvtColor(cropping, cv2.COLOR_BGR2RGB)

    # Add it to the database
    db.add_face(face_cropping=cropping, roll_number=roll_num, name=name)


def take_picture():
    cap = cv2.VideoCapture(0)
    image = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        # Display the resulting frame
        cv2.imshow('Webcam Video', frame)
        
        key = cv2.waitKey(1) & 0xFF  # Detect keypress

        # Check if the 'c' key is pressed to capture an image
        if key == ord('c'):
            print("Captured image")
            image = frame
            break
        
        # Check if the 'q' key is pressed to exit
        elif key == ord('q'):
            if image is None: image = frame
            break

    cap.release()
    cv2.destroyAllWindows()

    return image


if __name__ == '__main__':
    # Test to creation of profile
    create_profile()