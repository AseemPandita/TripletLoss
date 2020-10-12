from imutils import paths
import face_recognition
import pickle
import cv2
import os
from PIL import Image
import numpy as np

imagePaths = list(paths.list_files('yalefaces/train'))

knownNames = []
knownEncodings = []

for (i, imagePath) in enumerate(imagePaths):
    name = imagePath.split(os.path.sep)[-2]
    image_pil = Image.open(imagePath).convert('L')
    # Convert the image format into numpy array
    image = np.array(image_pil, 'uint8')
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    boxes = face_recognition.face_locations(rgb, model='cnn')

    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)


data = {'encodings':knownEncodings, 'names':knownNames}

f = open('encodings','wb')
f.write(pickle.dumps(data))
f.close()
    