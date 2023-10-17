import cv2


capture = cv2.VideoCapture(0)# Membuka sumber video dari kamera

while True:
    _, frame = capture.read()# Membaca frame dari sumber video

    # Menerapkan peningkatan kontras pada frame
    alpha = 2.0  # Faktor peningkatan kontras, Anda bisa mengubah nilainya sesuai kebutuhan
    beta = 50    # Nilai penambahan ke setiap piksel, Anda bisa mengubah nilainya sesuai kebutuhan
    contrasted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)  # Menerapkan peningkatan kontras pada frame

    cv2.imshow('Contrast-Enhanced Frame', contrasted)
    if cv2.waitKey(1) & 0xFF == ord('q'):# Menghentikan program ketika tombol 'q' ditekan
        break

capture.release()
cv2.destroyAllWindows()
