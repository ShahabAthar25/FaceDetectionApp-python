import cv2

# loading pre trained data here
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choosing face to detect on
img = cv2.imread('RDJ.jpg')