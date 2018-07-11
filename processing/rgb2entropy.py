import cv2, os, sys
import matplotlib.pyplot as plt

from os.path import basename
from skimage.filters.rank import entropy
from skimage.morphology import disk

filename = basename(sys.argv[1])
name, file_extension = os.path.splitext(filename)
output = os.path.join(sys.argv[2], filename)
output_vis = sys.argv[2] + "/" + name + "-visual" + file_extension

if(not os.path.isfile(output)):
    image = cv2.imread(sys.argv[1])
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #gray_equ = cv2.equalizeHist(gray)    
    entropy = entropy(gray, disk(5))  		    
    cv2.imwrite(output, entropy)
    plt.imsave(output_vis, entropy, cmap='viridis')    
else:
    print("entropy image " + filename + " already processed!")
