import cv2
import numpy as np

src =cv2.imread('catfish.jpg', 1)

# Menerapkan median Blur di src Image
dst = cv2.medianBlur(src,9)

# Menampilkan input dan output image
cv2.imshow("Median Smoothing",np.hstack((src, dst)))
cv2.waitKey(0) #Menunggu sampai keyboard ditekan
cv2.destroyAllWindows() #menghapus semua proses penampilan image