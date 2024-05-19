import cv2
import numpy as np
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
from scipy.spatial.distance import cosine
import os
# Enable MLIR and other related optimizations
os.environ['TF_MLIR_ENABLE_MLIR_BRIDGE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Enable XLA JIT compilation
tf.config.optimizer.set_jit(True)
# Load the pre-trained FaceNet model
facenet_model = tf.compat.v1.GraphDef()
with tf.io.gfile.GFile('facenet.pb', 'rb') as f:
    facenet_model.ParseFromString(f.read())

# Load the MTCNN detector
detector = MTCNN()


# Function to extract face embeddings
def get_embeddings(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)

    with tf.compat.v1.Session(graph=tf.compat.v1.Graph()) as sess:
        tf.import_graph_def(model, name='')
        embeddings = sess.run('embeddings:0', feed_dict={'input:0': samples, 'phase_train:0': False})
    return embeddings[0]  # Return 1-D embeddings instead of 2-D arrays


# Function to load images from a folder and extract embeddings
def load_embeddings_from_folder(folder):
    face_embeddings = []
    labels = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(img_rgb)
        if faces:
            x1, y1, width, height = faces[0]['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = img_rgb[y1:y2, x1:x2]
            face = cv2.resize(face, (160, 160))
            face_embedding = get_embeddings(facenet_model, face)
            face_embeddings.append(face_embedding)
            labels.append(filename.split('.')[0])
    return face_embeddings, labels


# Function to recognize faces from live camera feed
def recognize_faces_live(embeddings_frequency):
    cap = cv2.VideoCapture(0)
    frame_count = 0
    face_embeddings = []
    faces = []

    while True:
        ret, frame = cap.read()
        frame_count += 1

        if frame_count % embeddings_frequency == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(img_rgb)
            face_embeddings = []

            if faces:
                for face in faces:
                    x1, y1, width, height = face['box']
                    x1, y1 = abs(x1), abs(y1)
                    x2, y2 = x1 + width, y1 + height
                    face_img = img_rgb[y1:y2, x1:x2]
                    face_img = cv2.resize(face_img, (160, 160))
                    face_embedding = get_embeddings(facenet_model, face_img)
                    face_embeddings.append((face_embedding, (x1, y1, x2, y2)))

        if len(face_embeddings) > 0:
            # Compare face embeddings with known faces
            for face_embedding, (x1, y1, x2, y2) in face_embeddings:
                min_distance = 1
                identity = "Unknown"
                for known_embedding, label in zip(known_embeddings, labels):
                    distance = cosine(known_embedding, face_embedding)
                    if distance < min_distance:
                        min_distance = distance
                        identity = label # Threshold for recognition
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, identity, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



# Folder containing embeddings of known faces
dir = os.getcwd()
known_faces_folder = os.path.join(dir, 'faces')
known_embeddings, labels = load_embeddings_from_folder(known_faces_folder)

# Recognize faces from live camera feed every embeddings_frequency frames
recognize_faces_live(embeddings_frequency=16)
