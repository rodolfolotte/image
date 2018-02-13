import numpy as np
import cv2, os, sys

from matplotlib import pyplot as plt

from os.path import basename

from scipy import ndimage

from skimage import graph, data, io, segmentation, color, filters
from skimage.measure import regionprops
from skimage import draw

filename = basename(sys.argv[1])
name, file_extension = os.path.splitext(filename)

image = cv2.imread(sys.argv[1])
method = sys.argv[2]

if(method=="SLIC"):
	arg3 = int(sys.argv[3])
	arg4 = float(sys.argv[4])
	arg5 = int(sys.argv[5])
	output = name + "_" + sys.argv[2] + "_" + sys.argv[3] + "_" + sys.argv[4] + "_" + sys.argv[5] + file_extension
	segments = segmentation.slic(image, compactness=arg5, n_segments=arg3, sigma=arg4)
elif(method=="FELZENSZWALB"):
	arg3 = int(sys.argv[3])
	arg4 = float(sys.argv[4])
	arg5 = int(sys.argv[5])
	output = name + "_" + sys.argv[2] + "_" + sys.argv[3] + "_" + sys.argv[4] + "_" + sys.argv[5] + file_extension
	segments = segmentation.felzenszwalb(image, scale=arg3, sigma=arg4, min_size=arg5)
elif(method=="QUICKSHIFT"):
	arg3 = int(sys.argv[3])
	arg4 = int(sys.argv[4])
	arg5 = float(sys.argv[5])
	output = name + "_" + sys.argv[2] + "_" + sys.argv[3] + "_" + sys.argv[4] + "_" + sys.argv[5] + file_extension
	segments = segmentation.quickshift(image, kernel_size=arg3, max_dist=arg4, ratio=arg5)
elif(method=="WATERSHED"):
	arg3 = int(sys.argv[3])
	arg4 = float(sys.argv[4])
	output = name + "_" + sys.argv[2] + "_" + sys.argv[3] + "_" + sys.argv[4] + file_extension
	gray_image = color.rgb2gray(image)
	gradient = filters.sobel(gray_image)
	segments = segmentation.watershed(gradient, markers=arg3, compactness=arg4)
else:
    print("Wrong option to a method. The alternatives are SLIC, FELZENSZWALB, QUICKSHIFT, WATERSHED. Check it again")

segments = segments + 1
regions = regionprops(segments)
label_rgb = color.label2rgb(segments, image, kind='avg')
#label_rgb = segmentation.mark_boundaries(label_rgb, segments, (0, 0, 0))
final_label_rgb = color.label2rgb(label_rgb, image, kind='avg')

cv2.imwrite(output, final_label_rgb)