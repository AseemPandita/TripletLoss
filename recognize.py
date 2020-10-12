import cv2
import pickle
import face_recognition
from PIL import Image
import numpy as np


data = pickle.loads(open('encodings','rb').read())

image_pil = Image.open('yalefaces/test/subject03/subject03.centerlight').convert('L')
#image_pil = Image.open('a.png').convert('L')
image = np.array(image_pil, 'uint8')
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

boxes = face_recognition.face_locations(rgb, model='hog')
encodings = face_recognition.face_encodings(rgb, boxes)

names = []

for encoding in encodings:
    matches = face_recognition.compare_faces(data['encodings'], encoding, tolerance=0.6)
    name = 'unknown'

    if True in matches:
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}
            
        for i in matchedIdxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
        
        # update the list of names
        if counts[name] > 5:
            names.append(name)
        else:
            names.append('unknown')


for ((top, right, bottom, left), name) in zip(boxes, names):
	# draw the predicted face name on the image
	cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
	y = top - 15 if top - 15 > 15 else top + 15
	cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.75, (0, 255, 0), 2)
# show the output image

print('\n\n\n\n',names)


cv2.imshow("Image", image)
cv2.waitKey(0)