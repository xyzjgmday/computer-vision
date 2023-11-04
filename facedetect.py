import cv2

# Inisialisasi Cascade Classifier untuk wajah, mata, dan senyuman
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Buka gambar
img = cv2.imread('images.jpg', 1)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Deteksi wajah dalam gambar
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)

for (x_face, y_face, w_face, h_face) in faces:
    cv2.rectangle(img, (x_face, y_face), (x_face + w_face, y_face + h_face), (255, 130, 0), 2)
    roi_gray = gray_img[y_face:y_face + h_face, x_face:x_face + w_face]
    roi_color = img[y_face:y_face + h_face, x_face:x_face + w_face]

    # Deteksi mata dalam area wajah
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=18)
    for (x_eye, y_eye, w_eye, h_eye) in eyes:
        cv2.rectangle(roi_color, (x_eye, y_eye), (x_eye + w_eye, y_eye + h_eye), (0, 180, 60), 2)

    # Deteksi senyuman dalam area wajah
    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=2.5, minNeighbors=20)
    for (x_smile, y_smile, w_smile, h_smile) in smiles:
        cv2.rectangle(roi_color, (x_smile, y_smile), (x_smile + w_smile, y_smile + h_smile), (255, 0, 130), 2)

# Tampilkan gambar dengan deteksi wajah, mata, dan senyuman
cv2.imshow('Output', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
