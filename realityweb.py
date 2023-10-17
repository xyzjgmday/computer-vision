import cv2

capture = cv2.VideCapture(0)

while(True):
    _ , frame = capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', gray)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

capture.release
cv2.destroyAllWindows