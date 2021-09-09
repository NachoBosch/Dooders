import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('gender_dataset/train/women/00000008.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
print(faces)
for (x, y, w, h) in faces:
    print(x,y)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_face = img[x:x+w,y:y+h].copy()
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    gray = cv2.cvtColor(roi_face, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
    for (xi, yi, wi, hi) in eyes:
        cv2.circle(img, (int(xi), int(yi)), 10, (75,102,138), -1)

cv2.imshow('img', img)
cv2.waitKey(2000)
cv2.destroyAllWindows()
