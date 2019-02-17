import cv2
import numpy as np

from np_writer import f_name

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv(f_name).values

X, Y = data[:, 1:-1], data[:, -1]

print(X.shape, Y.shape)

model = KNeighborsClassifier(n_neighbors=4)

model.fit(X, Y)

cap = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier("../../dataset/haarcascade_frontalface_default.xml")

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = classifier.detectMultiScale(gray, 1.5, 5)

    X_test = []

    for face in faces:
        x, y, w, h = face
        im_face = gray[y:y+h, x:x+w]
        im_face = cv2.resize(im_face, (100, 100))
        X_test.append(im_face.reshape(-1))
        # cv2.imshow("face", im_face)

    if len(faces) > 0:
        response = model.predict(np.array(X_test))

        for i, face in enumerate(faces):
            x, y, w, h = face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 5)
            cv2.putText(frame, response[i], (x-50, y-50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)

    # if not ret:
    #     continue

    cv2.imshow("full", frame)

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break



# np.save(name, np.array(f_list))

cap.release()
