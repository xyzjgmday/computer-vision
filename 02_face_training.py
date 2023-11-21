import cv2
import numpy as np
from PIL import Image
import os

# Path untuk database gambar wajah
path = 'dataset'

# Inisialisasi LBPH Face Recognizer dan Cascade Classifier
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # Fix the filename

# Fungsi untuk mendapatkan data gambar dan label
def getImageAndLabels(path):
    # Mendapatkan path gambar dari folder dataset
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    # Loop untuk setiap path gambar
    for imagePath in imagePaths:
        # Membuka gambar dan mengonversinya ke grayscale
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')

        # Mendapatkan ID wajah dari nama file
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        # Loop untuk setiap wajah yang terdeteksi
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    return faceSamples, ids

# Menampilkan pesan sebelum proses pelatihan dimulai
print("\n [INFO] Training Faces. Ini akan memakan waktu sekejap saja, tungguin benar ya. . .")

# Mendapatkan data gambar dan label
faces, ids = getImageAndLabels(path)

# Melatih recognizer dengan data gambar dan label
recognizer.train(faces, np.array(ids))

# Menyimpan model pelatihan ke dalam file 'trainer/trainer.yml'
recognizer.write('trainer/trainer.yml')

# Menampilkan pesan setelah proses pelatihan selesai
print("\n [INFO] {0} Latihan muka beres, keluar dari program".format(len(np.unique(ids))))
