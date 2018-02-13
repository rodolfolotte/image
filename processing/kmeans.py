import numpy as np
import cv2, os, sys
import matplotlib.pyplot as plt

from os.path import basename

filename = basename(sys.argv[1])
name, file_extension = os.path.splitext(filename)
output = name + "_" + sys.argv[2] + file_extension

img = cv2.imread(sys.argv[1])
K = int(sys.argv[2])
Z = img.reshape((-1,3))
Z = np.float32(Z)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

gray_img = cv2.cvtColor(res2, cv2.COLOR_RGB2GRAY)
cv2.imwrite(output, gray_img)