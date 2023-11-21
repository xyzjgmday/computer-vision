import cv2

# Inisialisasi kamera
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set lebar video
cam.set(4, 480)  # set tinggi video

# Inisialisasi detektor wajah menggunakan cascade classifier
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Meminta input user untuk ID wajah
face_id = input('\n Masukkan user id lalu tekan <return> ==>    ')

# Menampilkan pesan awal
print("\n [INFO] Initializing face capture. Lihatlah kamera dan rasakan ketampanan anda . . .")

# Inisialisasi variabel count untuk menyimpan jumlah sampel wajah
count = 0

# Loop utama untuk mengambil sampel wajah
while True:
    # Membaca frame dari kamera
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Mendeteksi wajah dalam frame
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # Loop untuk setiap wajah yang terdeteksi
    for (x, y, w, h) in faces:
        # Menggambar kotak di sekitar wajah
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Menyimpan gambar wajah yang terdeteksi ke folder dataset
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

        # Menampilkan frame dengan kotak wajah
        cv2.imshow('image', img)

    # Menunggu tombol k pada keyboard, 'ESC' untuk keluar dari aplikasi
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 30:  # Mengambil 30 sampel lalu menghentikan video
        break

# Membersihkan dan menutup program
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
