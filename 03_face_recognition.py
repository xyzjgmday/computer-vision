import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

# Inisialisasi counter id
id = 0

# Nama yang berkaitan dengan id: contoh ==> Marcelo: id=1, dll
names = ['None', 'Day', 'Nurhidayat', 'xyzjgmday', 'D', 'N']

# Inisialisasi dan mulai tangkapan video secara real-time
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # atur lebar video
cam.set(4, 480)  # atur tinggi video

# Tentukan ukuran jendela minimum untuk dikenali sebagai wajah
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        # Periksa jika confidence kurang dari 100 ==> "0" adalah kecocokan yang sempurna
        if confidence < 100:
            id = names[id]
            confidence = " {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = " {0}%".format(round(100 - confidence))
        cv2.putText(img, str(id), (x + 5, y + 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5),
                    font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)
    k = cv2.waitKey(10) & 0xFF  # Tekan 'q' untuk keluar
    if k == ord('q'):  # Tekan tombol 'q' untuk keluar
        break

# Setelah keluar dari loop, lakukan pembersihan
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
