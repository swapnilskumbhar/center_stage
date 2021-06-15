import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

thresh = 100
while True:
    _, img = cap.read()
    ht,wd = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    n_faces = len(faces)
    if n_faces > 0:
        x, y, w, h = faces[0]
        x1,y1,x2,y2 = x,y, x+w, y+h
    if n_faces > 1:
        faces1 = np.array(faces)
        faces1[:,2] += faces1[:,0]  
        faces1[:,3] += faces1[:,1] 
        x1 = np.min(faces1,axis=0)
        y1 = x1[1]
        x1 = x1[0]
        x2 = np.max(faces1,axis=0)
        y2 = x2[3]
        x2 = x2[2]
        
    crp_c0 = x1 - thresh if x1-thresh > 0 else 0
    crp_c1 = x2 + thresh if x2+thresh < wd else wd
    crp_r0 = y1 - thresh if y1-thresh > 0 else 0
    crp_r1 = y2 + thresh if y2+thresh < ht else ht
    
    cv2.imshow('center stage', img[crp_r0:crp_r1, crp_c0:crp_c1])
    cv2.imshow('actual camera footage', img)
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

cap.release()