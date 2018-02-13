import numpy as np
import cv2, os, sys
from matplotlib import pyplot as plt

from os.path import basename

filename = basename(sys.argv[1])
name, file_extension = os.path.splitext(filename)
output = name + "_" + sys.argv[2] + file_extension

img = cv2.imread(sys.argv[1])

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()

# find and draw the keypoints
kp = fast.detect(img, None)
img2 = cv2.drawKeypoints(img, kp, 0, color=(0,255,0), flags=0)

cv2.imwrite(output, img2)

# Disable nonmaxSuppression
fast.setBool('nonmaxSuppression', 0)
kp = fast.detect(img,None)

print("Total Keypoints without nonmaxSuppression: ", len(kp))

img3 = cv2.drawKeypoints(img, kp, 0, color=(0,255,0), flags=0)

cv2.imwrite(output, img3)