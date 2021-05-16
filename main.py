import cv2

# loading pre trained data here
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choosing face to detect on
img = cv2.imread('RDJ.jpg')

# Turning Img To Grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cordinates = trained_face_data.detectMultiScale(grayscaled_img)

print(face_cordinates)

# Showing Image
cv2.imshow('My Face: ', grayscaled_img)

# listening for key press
cv2.waitKey()