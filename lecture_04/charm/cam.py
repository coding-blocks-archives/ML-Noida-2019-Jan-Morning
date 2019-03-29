import cv2

cap = cv2.VideoCapture(0)

detector = cv2.CascadeClassifier("../../dataset/haarcascade_frontalface_default.xml")

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    faces = detector.detectMultiScale(frame, 1.5, 3)

    faces = faces[:1]

    for face in faces:
        x, y, w, h = face

        only_face = frame[y:y+h, x:x+w]
        cv2.imshow("face", only_face)

    cv2.imshow("camera", frame)

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('c'):
        cv2.imwrite("class.png", frame)

cap.release()

