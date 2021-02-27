import cv2
import face_recognition

cam = cv2.VideoCapture(0)

image = face_recognition.load_image_file("tst.jpeg")
encoding = face_recognition.face_encodings(image)[0]
known_face = [encoding]

while True:
    ret, frame = cam.read()

    face_location = face_recognition.face_locations(frame)
    face_encoding = face_recognition.face_encodings(frame, face_location)

    cv2.imshow("Detecting...", frame)

    for encode in face_encoding:
        matches = face_recognition.compare_faces(known_face, encode)
        distance = face_recognition.face_distance(known_face, encode)
        if (distance < 0.4):
            print("success")
            cam.release()
