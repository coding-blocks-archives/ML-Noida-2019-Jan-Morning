import cv2
import numpy as np

import np_writer

name = input("Enter your name : ")

cap = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier("../../dataset/haarcascade_frontalface_default.xml")

f_list = []

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = classifier.detectMultiScale(gray, 1.5, 5)

    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)

    faces = faces[:1]

    if len(faces) == 1:
        face = faces[0]
        x, y, w, h = face
        im_face = frame[y:y+h, x:x+w]
        cv2.imshow("face", im_face)


    # if not ret:
    #     continue

    cv2.imshow("full", frame)

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('c'):
        if len(faces) == 1:
            gray_face = cv2.cvtColor(im_face, cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (100, 100))
            print(len(f_list), type(gray_face), gray_face.shape)
            f_list.append(gray_face.reshape(-1))
        else:
            print("face not found")

        if len(f_list) == 10:
            break


np_writer.write(name, np.array(f_list))

# np.save(name, np.array(f_list))

cap.release()
