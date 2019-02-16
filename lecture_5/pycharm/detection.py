import cv2

cap = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier("../../dataset/haarcascade_frontalface_default.xml")

while True:

    ret, frame = cap.read()

    faces = classifier.detectMultiScale(frame, 1.5, 5)

    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)

    faces = faces[:1]

    if len(faces) == 1:
        face = faces[0]
        x, y, w, h = face
        im_face = frame[y:y+h, x:x+w]
        cv2.imshow("face", im_face)


    print(len(faces))

    # if not ret:
    #     continue

    cv2.imshow("full", frame)

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break


cap.release()
