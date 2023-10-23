import cv2

# Inisialisasi kamera
capture = cv2.VideoCapture(0)

# Load classifier wajah
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    # Membaca frame dari kamera
    ret, frame = capture.read()

    # Mengonversi frame ke grayscale (untuk deteksi wajah)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Mendeteksi wajah dalam frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.5, minNeighbors=5)

    # Menggambar kotak di sekitar wajah yang terdeteksi
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Menampilkan frame dengan kotak wajah
    cv2.imshow('Face Detection', frame)

    # Menghentikan program saat tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Melepaskan kamera dan menutup semua jendela tampilan
capture.release()
cv2.destroyAllWindows()
