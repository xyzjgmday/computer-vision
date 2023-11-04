import cv2
import numpy as np

src =cv2.imread('snail.jpg', cv2.IMREAD_UNCHANGED)

# Menerapkan Gaussian Blur di src Image
dst = cv2.GaussianBlur(src,(5,5),10)

# Menampilkan input dan output image
cv2.imshow("Gausian Smoothing",np.hstack((src, dst)))
cv2.waitKey(0) #Menunggu sampai keyboard ditekan
cv2.destroyAllWindows() #menghapus semua proses penampilan image