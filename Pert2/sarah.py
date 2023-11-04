import cv2

img = cv2.imread('anakkucing.jpg', 0)

cv2.imshow('Gambar Kucing ', img)

cv2.waitKey(0)

cv2.destroyAllWindows()
