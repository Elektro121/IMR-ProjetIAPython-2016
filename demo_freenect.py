#!/usr/bin/env python
from freenect import sync_get_depth as get_depth, sync_get_video as get_video
import cv2 as cv  
import numpy as np
# -*- coding: utf-8 -*-
  
faceCascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
eyeCascade = cv.CascadeClassifier('haarcascade_eye.xml')
smileCascade = cv.CascadeClassifier('haarcascade_smile.xml')

def doloop():
	global depth, rgb
	while True:
		# Get a fresh frame
		(depth,_), (rgb,_) = get_depth(), get_video()
		(frame,_) = (rgb, _)		
		
		gray = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)
		
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
		
		# Display the resulting frame
		cv.imshow('Video', frame)
		
		# Build a two panel color image
		d3 = np.dstack((depth,depth,depth)).astype(np.uint8)
		da = np.hstack((d3,rgb))
		
		# Simple Downsample
		cv.imshow('both',np.array(da[::2,::2,::-1]))
		cv.waitKey(5)

doloop()

"""
IPython usage:
 ipython
 [1]: run -i demo_freenect
 #<ctrl -c>  (to interrupt the loop)
 [2]: %timeit -n100 get_depth(), get_rgb() # profile the kinect capture

"""

