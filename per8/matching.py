import cv2
import numpy as np

MIN_MATCH_COUNT = 4

imgname1 = "template.png"
imgname2 = "find.png"

# (1) persiapan data
img1 = cv2.imread(imgname1)
img2 = cv2.imread(imgname2)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# (2) Membuat SIFT Objek
sift = cv2.SIFT_create()

# (3) Membuat flann matcher
matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {})

# (4) mencari keypoint dan deskripsi dari image 1 dan image 2
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# (5) menggunakan matcher untuk mencocokkan deskriptor
matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
# Sortir Jarak
matches = sorted(matches, key=lambda x: x[0].distance)

# (6) filter matches based on Lowe's ratio test
good = [m1 for (m1, m2) in matches if m1.distance < 0.7 * m2.distance]
canvas = img2.copy()

# (7) Mencari Homography Matrix
if len(good) > MIN_MATCH_COUNT:
    # (queryIdx untuk objek kecil, trainIdx untuk Scene)
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    # MEncari Homography matrix di cv2.RANSAC menggunakan good match pouint
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    h, w = img1.shape[:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    cv2.polylines(canvas, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))

# (8) drawMatches
matched = cv2.drawMatches(img1, keypoints1, canvas, keypoints2, good, None)

# (9) Crop the matched region from the scene
h, w = img1.shape[:2]
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, M)
perspectiveM = cv2.getPerspectiveTransform(np.float32(dst), pts)
found = cv2.warpPerspective(img2, perspectiveM, (w, h))

# (10) save and display
cv2.imwrite("matched.png", matched)
cv2.imwrite("found.png", found)
cv2.imshow("matched", matched)
cv2.imshow("found", found)
cv2.waitKey()
cv2.destroyAllWindows()
