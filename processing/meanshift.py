import numpy as np
import cv2, os, sys
import matplotlib.pyplot as plt

from skimage.transform import rescale

from os.path import basename

filename = basename(sys.argv[1])
name, file_extension = os.path.splitext(filename)
output = name + "_" + sys.argv[2] + file_extension
scale_factor = float(sys.argv[2])
hs = int(sys.argv[3])
hr = int(sys.argv[4])

image = cv2.imread(sys.argv[1])
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
Z = np.float32(image.reshape((-1,3)))

# Python: cv2.pyrMeanShiftFiltering(src, sp, sr[, dst[, maxLevel[, termcrit]]])
	# src – The source 8-bit, 3-channel image.
	# dst – The destination image of the same format and the same size as the source.
	# sp – The spatial window radius.
	# sr – The color window radius.
	# maxLevel – Maximum level of the pyramid for the segmentation.
	# termcrit – Termination criteria: when to stop meanshift iterations.
image = cv2.pyrMeanShiftFiltering(image, hs, hr, 2)
image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

plt.imshow(image)
plt.show()