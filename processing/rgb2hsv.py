import cv2, os, sys
import matplotlib.pyplot as plt

from os.path import basename

filename = basename(sys.argv[1])
output = os.path.join(sys.argv[2], filename)

if(not os.path.isfile(output)):
    image = cv2.imread(sys.argv[1])
    #gray_equ = cv2.equalizeHist(rgb_gray) 
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    cv2.imwrite(output, hsv)
else:
    print("hsv image " + filename + " already processed!")