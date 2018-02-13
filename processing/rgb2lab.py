import cv2, os, io, sys
    
from os.path import basename
from skimage import io, color

filename = basename(sys.argv[1])
output = os.path.join(sys.argv[2], filename)

if(not os.path.isfile(output)):
    image = cv2.imread(sys.argv[1])
    lab = color.rgb2lab(image)
    #gray_equ = cv2.equalizeHist(rgb_gray)    
    cv2.imwrite(output, lab)    
else:
    print("lab image " + filename + " already processed!")