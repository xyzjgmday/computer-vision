import cv2
import numpy as np

src =cv2.imread('goat.jpg', 1)

# Menerapkan averaging Blur di src Image
dst = cv2.blur(src,(5,25))


# Menampilkan input dan output image
cv2.imshow("Averaging Smoothing",np.hstack((src, dst)))
cv2.waitKey(0) #Menunggu sampai keyboard ditekan
cv2.destroyAllWindows() #menghapus semua proses penampilan image