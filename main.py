import cv2
from random import randrange

# loading pre trained data here
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choosing face to detect on
img = cv2.imread('RDJ.jpg')

# Turning Img To Grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Getting Face Cords
face_cordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Adding Rectangle On Faces
for (x, y, w, h) in face_cordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(255), randrange(255), randrange(255)), 2)

# Showing Image
cv2.imshow('My Face: ', img)

# listening for key press
cv2.waitKey()