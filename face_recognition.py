import cv2
import face_recognition
import numpy as np

imgElon =face_recognition.load_image_file('image/ellon-1.jpeg')
imgElon=cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest =face_recognition.load_image_file('image/ellon-2.jpeg')
imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

facloc=face_recognition.face_locations(imgElon)[0]
encodElon=face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(facloc[3],facloc[0]),(facloc[1],facloc[2]),(255,0,255),2)

facloc1=face_recognition.face_locations(imgTest)[0]
encodElon1=face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(facloc1[3],facloc1[0]),(facloc1[1],facloc1[2]),(255,0,255),2)

results=face_recognition.compare_faces([encodElon],encodElon1)
facedis=face_recognition.face_distance([encodElon],encodElon1)
print(results,facedis)
cv2.imshow('elon img',imgElon)
cv2.imshow('elon test',imgTest)
cv2.waitKey(0)