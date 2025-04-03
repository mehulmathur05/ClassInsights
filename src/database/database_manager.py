import chromadb
import insightface
import numpy as np
import os
import cv2

class ImageDatabase:
    def __init__(self, persistence_path=None):
        # Default initializer of persistance path
        if persistence_path is None:
            project_root = os.path.dirname(os.path.abspath(__file__))
            persistence_path = os.path.join(project_root, '../../tests/data/chromadb')

        # Ensure that the path where the database exists or will be created exists
        os.makedirs(persistence_path, exist_ok=True)

        # Set up ChromaDB client and create or get the "faces" collection
        self.chroma_client = chromadb.PersistentClient(path=persistence_path)
        self.collection = self.chroma_client.get_or_create_collection(name="faces")
        
        # Initialize InsightFace model for generating face embeddings
        self.face_model = insightface.app.FaceAnalysis()
        self.face_model.prepare(ctx_id=-1)  # change to ctx_id=0 for GPU


    # Add the embeddings of a face cropping to the database
    def add_face(self, face_cropping: np.ndarray, roll_number: str, name: str) -> None:
        faces = self.face_model.get(face_cropping)

        if len(faces) != 1:
            raise ValueError(f"Expected exactly one face in the cropping, but found {len(faces)}")
        embedding = faces[0].embedding.tolist()

        # Store the embedding in ChromaDB with the roll number as the ID, and the name as metadata
        self.collection.add(
            embeddings=[embedding],
            ids=[roll_number],
            metadatas=[{"roll_number": roll_number, "name": name}]
        )


    # Query the most similar face
    def query_face(self, face_cropping: np.ndarray) -> str:
        faces = self.face_model.get(face_cropping)  # Generate the embeddings

        if len(faces) != 1:
            print(f"Expected exactly one face in the cropping, but found {len(faces)}")
            return None, None
        
        query_embedding = faces[0].embedding.tolist()  # ChromaDB requires it to be a list and not ndarray

        # Query the most similar embedding
        results = self.collection.query(query_embeddings=[query_embedding], n_results=1)

        # Return the serial number of the closest match, or None if no matches exist
        if results["ids"][0]:
            return results["ids"][0][0], results['metadatas'][0][0]['name']
        
        return None, None
    
    
    # Remove a collection from the database
    def remove_collection(self, collection):
        self.chroma_client.delete_collection(name=collection)
        print(f"The collection {collection} was successfully removed")


if __name__ == '__main__':
    # Initialize relevant directory and file paths
    parent_path = os.path.dirname(os.path.abspath(__file__))
    source_path = os.path.dirname(parent_path)
    base_path = os.path.dirname(source_path)
    croppings_dir = os.path.join(base_path, 'results/croppings/')
    test_data_dir = os.path.join(base_path, 'tests/data')
    database_directory = os.path.join(base_path, 'tests', 'data', 'chromadb')

    # Initialize the ImageDatabase object
    db = ImageDatabase(database_directory)

    # Add each cropping in the cropping directory
    for filename in os.listdir(croppings_dir):
        cropping = cv2.imread(os.path.join(croppings_dir, filename))
        cropping = np.array(cv2.cvtColor(cropping, cv2.COLOR_BGR2RGB))
        filename = os.path.splitext(filename)[0]
        roll_number = filename.split('_')[1]
        db.add_face(cropping, roll_number, filename)

    test_name = 'chandler'
    test_image = cv2.imread(os.path.join(test_data_dir, test_name + '.png'))
    test_image = np.array(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
        
    match_id, match_name = db.query_face(test_image)
    print(f"The image best matching to {test_name} is {match_name} with id {match_id}")

    # breakpoint()