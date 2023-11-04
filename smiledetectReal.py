import cv2

# Inisialisasi Cascade Classifier untuk wajah, mata, dan senyuman
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah dalam frame
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)

    for (x_face, y_face, w_face, h_face) in faces:
        cv2.rectangle(frame, (x_face, y_face), (x_face + w_face, y_face + h_face), (255, 130, 0), 2)
        roi_gray = gray_img[y_face:y_face + h_face, x_face:x_face + w_face]
        roi_color = frame[y_face:y_face + h_face, x_face:x_face + w_face]

        # Deteksi mata dalam area wajah
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=18)
        for (x_eye, y_eye, w_eye, h_eye) in eyes:
            cv2.rectangle(roi_color, (x_eye, y_eye), (x_eye + w_eye, y_eye + h_eye), (0, 180, 60), 2)

        # Deteksi senyuman dalam area wajah
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=2.5, minNeighbors=20)
        for (x_smile, y_smile, w_smile, h_smile) in smiles:
            cv2.rectangle(roi_color, (x_smile, y_smile), (x_smile + w_smile, y_smile + h_smile), (255, 0, 130), 2)

    # Tampilkan frame dengan deteksi wajah, mata, dan senyuman
    cv2.imshow('Face and Smile Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Tekan tombol 'Esc' untuk keluar
        break

cap.release()
cv2.destroyAllWindows()
