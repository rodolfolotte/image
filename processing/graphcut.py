import numpy as np
import cv2, os, sys

from matplotlib import pyplot as plt

from os.path import basename

from scipy import ndimage

from skimage import draw, data, io, segmentation, color
from skimage.future import graph
from skimage.measure import regionprops


def display_edges(image, g, threshold):
    """Draw edges of a RAG on its image
 
    Returns a modified image with the edges drawn.Edges are drawn in green
    and nodes are drawn in yellow.
 
    Parameters
    ----------
    image : ndarray
        The image to be drawn on.
    g : RAG
        The Region Adjacency Graph.
    threshold : float
        Only edges in `g` below `threshold` are drawn.
 
    Returns:
    out: ndarray
        Image with the edges drawn.
    """
    image = image.copy()
    for edge in g.edges_iter():
        n1, n2 = edge
 
        r1, c1 = map(int, rag.node[n1]['centroid'])
        r2, c2 = map(int, rag.node[n2]['centroid'])
 
        line  = draw.line(r1, c1, r2, c2)
        circle = draw.circle(r1,c1,2)
 
        if g[n1][n2]['weight'] < threshold :
            image[line] = 0,1,0
            
        image[circle] = 1,1,0
 
    return image


filename = basename(sys.argv[1])
name, file_extension = os.path.splitext(filename)

image = cv2.imread(sys.argv[1])
method = sys.argv[2]
cuts = int(sys.argv[6])

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
label_rgb = color.label2rgb(label_rgb, image, kind='avg')

rag = graph.rag_mean_color(image, segments)

for region in regions:
    rag.node[region['label']]['centroid'] = region['centroid']

#SHOW EDGES
edges_drawn_all = display_edges(label_rgb, rag, cuts)

#final_labels = graph.cut_threshold(segments, rag, cuts)
#final_label_rgb = color.label2rgb(final_labels, image, kind='avg')

cv2.imwrite(output, edges_drawn_all)

# fig = plt.figure("Superpixels -- %d segments" % (num_segments))
# ax = fig.add_subplot(1,1,1)
# ax.imshow(edges_drawn_all)
# plt.axis("off")
# plt.show()