import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
# Path to the folder containing images
path = 'image'
images = []
classNames = []
myList = os.listdir(path)

# Load images and extract class names (names of people)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Function to encode the faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+')as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dtString=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')



    
# Get the encodings for the known images
encodeListKnown = findEncodings(images)
print(f"Number of encodings found: {len(encodeListKnown)}")

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Find face locations and encodings in the current frame
    facloc = face_recognition.face_locations(imgS)
    encodElon = face_recognition.face_encodings(imgS, facloc)

    for encodeface, faceloc in zip(encodElon, facloc):
        # Compare the face encodings with the known encodings
        matches = face_recognition.compare_faces(encodeListKnown, encodeface)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeface)
        print(faceDis)
        matchesInd = np.argmin(faceDis)

        if matches[matchesInd]:
            name = classNames[matchesInd].upper()
            print(name)
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    # Display the resulting image in a window
    cv2.imshow('Webcam', img)

    # Check if 'q' is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

# facloc=face_recognition.face_locations(imgElon)[0]
# encodElon=face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon,(facloc[3],facloc[0]),(facloc[1],facloc[2]),(255,0,255),2)

# facloc1=face_recognition.face_locations(imgTest)[0]
# encodElon1=face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest,(facloc1[3],facloc1[0]),(facloc1[1],facloc1[2]),(255,0,255),2)

# results=face_recognition.compare_faces([encodElon],encodElon1)
# facedis=face_recognition.face_distance([encodElon],encodElon1)

