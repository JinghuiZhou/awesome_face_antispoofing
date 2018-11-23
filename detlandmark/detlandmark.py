import face_alignment
from skimage import io
import os
import sys
import glob
import cv2
import pickle
imgdir = sys.argv[1]
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
input = io.imread(imgdir)
preds = fa.get_landmarks(input)

for pred in preds:
	img = cv2.imread(imgdir)
	print('ldmk num:', pred.shape[0])
	for i in range(pred.shape[0]):
		x,y = pred[i]
		print(x,y)
		cv2.circle(img,(x,y),1,(0,0,255),-1)
	cv2.imshow('-',img)
	cv2.waitKey()
