#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from freenect import sync_get_depth as get_depth, sync_get_video as get_video
import freenect
import cv2 as cv  
import numpy as np
import dlib
import math

  
faceCascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
eyeCascade = cv.CascadeClassifier('haarcascade_eye.xml')
smileCascade = cv.CascadeClassifier('haarcascade_smile.xml')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
win = dlib.image_window()


global tilt, depth, rgb, gray, frame, tab_points, face_view, viewport_face
global AUtable

viewport_face = 300

def doloop():
    global depth, rgb, gray, frame, roi_color, roi_gray
    global tilt
    tilt = 0
    while True:
        # Get a fresh frame
        (depth,_), (rgb,_) = get_depth(), get_video()
        (frame,_) = (rgb, _)        
        
        gray = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)

        landmarkDetection()
        #faceDetection()
        # Display the resulting frame
        #cv.imshow('Video', frame)
        
        # Build a two panel color image
        d3 = np.dstack((depth,depth,depth)).astype(np.uint8)
        da = np.hstack((d3,rgb))
        
        # Simple Downsample
        #cv.imshow('both',np.array(da[::2,::2,::-1]))
        cv.imshow('vue', rgb)
        keyPressed = cv.waitKey(5)
        #print(keyPressed)
        if(keyPressed == 1048673) : #a
            tilt = tilt + 10
            freenect.set_tilt_degs(dev, tilt)
        if (keyPressed == 1048689):  #q
            tilt = tilt - 10
            freenect.set_tilt_degs(dev, tilt)
        if (keyPressed == 1048698): #z
            break;

def faceDetection():
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        #On met un rectangle autour du visage
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #On récupère l'image en question
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = rgb[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(
                            roi_gray,
                            scaleFactor=1.1,
                            minNeighbors=10,
                            minSize=(30, 30),
                            flags=cv.CASCADE_SCALE_IMAGE
                            )
        smiles = smileCascade.detectMultiScale(
                            roi_gray,
                            scaleFactor=1.1,
                            minNeighbors=40,
                            minSize=(30, 30),
                            flags=cv.CASCADE_SCALE_IMAGE
                            )
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
        for (ex,ey,ew,eh) in smiles:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
            break;

def landmarkDetection():
    global tab_points
    tab_points = []
    #win.clear_overlay()
    #win.set_image(frame)
    dets = detector(frame, 1)
    #print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Width: {} Height:{}".format(k, d.left(), d.top(), d.right(), d.bottom(), d.width(), d.height()))
        cv.rectangle(frame, (d.left(), d.top()), (d.left() + d.width(), d.top() + d.height()), (0, 255, 0), 2)
        # Get the landmarks/parts for the face in box d
        shape = predictor(frame, d)

        face_view = np.zeros((viewport_face, viewport_face, 3), np.uint8)

        #scale_x = float(viewport_face)/d.width() - 1.2
        #scale_y = float(viewport_face)/d.height() - 1.2
        #scale_x = 0.8 if (float(viewport_face)/d.width() < 1) else scale_x
        #scale_y = 0.8 if (float(viewport_face)/d.height() < 1) else scale_y

        face_width = int(((shape.part(16).x)-(shape.part(0).x))/2)
        face_height = int((shape.part(8).y))-((max(shape.part(16).y,shape.part(24).y)))
        scale_x = float(viewport_face)/(face_width*4)
        scale_y = float(viewport_face)/(face_height*2)

        #normalisation_x = 1/shapes[]
        #normaisation_y =
        #print((scale_x, scale_y))

        #print("Nb of points : {}", format(shape.num_parts))
        #print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
        #                                          shape.part(1)))
        i = 0
        color = (0,0,0)
        for p in shape.parts():
            showPoints(p, i)
            #print(p)
            makeFacePoints(p, i, shape, scale_x, scale_y, tab_points, face_view)
            i = i + 1
        # Draw the face landmarks on the screen.
        # win.add_overlay(shape)
        cv.circle(face_view,
                  (
                      int((tab_points[38][0]+tab_points[37][0])/2),
                      int((tab_points[37][1]+tab_points[41][1])/2)
                  ),
                   3,
                   (255,255,255))
        cv.circle(face_view,
                  (
                      int((tab_points[44][0] + tab_points[43][0]) / 2),
                      int((tab_points[43][1] + tab_points[47][1]) / 2)
                  ),
                  3,
                  (255, 255, 255))
        cv.imshow('face_norm', face_view)
        AUCalculation()
        emotionDetector()
    #win.add_overlay(dets)
    #dlib.hit_enter_to_continue()

def showPoints(p, i):
    if (i % 2 == 0):
        color = (0, 255, 255)
    else:
        color = (255, 255, 0)
    cv.circle(rgb, (p.x, p.y), 1, color)
    cv.putText(rgb,
               str(i),
               (p.x, p.y),
               cv.FONT_HERSHEY_COMPLEX_SMALL,
               0.5,
               (255, 255, 255)
               )

def makeFacePoints(p, i, shape, scale_x, scale_y, tab_points, face_view):
    # pos_norm_x = int(math.floor((p.x-d.left())*scale_x))
    # pos_norm_y = int(math.floor((p.y-d.top())*scale_y))
    pos_norm_x = int(math.floor((p.x - shape.parts()[30].x) * scale_x) + viewport_face / 2)
    pos_norm_y = int(math.floor((p.y - shape.parts()[30].y) * scale_y) + viewport_face / 2)
    pos_norm_x = pos_norm_x if (pos_norm_x < viewport_face - 1) else viewport_face - 1
    pos_norm_y = pos_norm_y if (pos_norm_y < viewport_face - 1) else viewport_face - 1
    tab_points.append((
        pos_norm_x,
        pos_norm_y
    ))
    #print(tab_points[i])
    color = (255, 255, 255)
    if(i == 48 or i == 54 or i == 30):
        cv.circle(face_view, tab_points[i], 2, color)
        face_view[tab_points[i][1], tab_points[i][0]] = color
    else:
        face_view[tab_points[i][1], tab_points[i][0]] = color

def AUCalculation():
    global AUtable
    AUtable = [False for i in range(64)]
    #AUtable[0:64] = False

    # 6 : Cheek Raiser

    # 12 : Lip Corner Puller
    if (
            max(tab_points[50][1], tab_points[51][1]) > max(tab_points[54][1],tab_points[48][1])
    ):
        AUtable[12]= True;

    # 26 : Jaw drop


def emotionDetector():
    global AUtable
    emotion = ""
    if(AUtable[12]):
        emotion = "smile"
    else:
        emotion = "???"
    print(emotion)
    cv.putText(rgb,
               emotion,
               (450, 50),
               cv.FONT_HERSHEY_SIMPLEX,
               2,
               (255, 255, 255)
               )




ctx = freenect.init()
dev = freenect.open_device(ctx, freenect.num_devices(ctx))
doloop()
freenect.stop_depth(dev)
freenect.stop_video(dev)
freenect.close_device(dev)

"""
IPython usage:
 ipython
 [1]: run -i demo_freenect
 #<ctrl -c>  (to interrupt the loop)
 [2]: %timeit -n100 get_depth(), get_rgb() # profile the kinect capture

"""

