import cv2

# All the faces that were already trained; helps to be more accurate
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# User Image
img = cv2.imread("2_faces.jpg")

# Turning user image into gray scale (gray and black - gets rid of all colors)
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Finds the coordinates of where the face is (top left and bottom right)
face_coordinates = list(trained_face_data.detectMultiScale(grayscaled_img))

# Checks to see if there is more than 1 face in the image
if len(face_coordinates) > 1:
    # If the above condition is true, both are detected and a rectangle is drawn on both
    for coordinate_array in face_coordinates:
        x, y, w, h = coordinate_array
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
else:
    x, y, w, h = face_coordinates
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Displays the image with rectangle on top
cv2.imshow("Face Detector - Python + OpenCV", img)
cv2.waitKey()

# Confirmation
print("Code Completed, No Problems Detected")