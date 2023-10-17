import cv2


capture = cv2.VideoCapture(0)# Membuka sumber video dari kamera

while True:
    _, frame = capture.read()# Membaca frame dari sumber video

    # Terapkan filter median pada frame
    median_filtered = cv2.medianBlur(frame, 5)  # Ubah nilai kernel

    cv2.imshow('Median Filtered Frame', median_filtered)  # Menampilkan frame yang telah dihaluskan menggunakan filter median

    if cv2.waitKey(1) & 0xFF == ord('q'):# Menghentikan program ketika tombol 'q' ditekan
        break

capture.release()
cv2.destroyAllWindows()
