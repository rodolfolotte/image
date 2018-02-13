# Image processing

Easy to use command-line image processing. Approaches using Python 3.5.2.

## Usage
Those methods operate independently. This allows easy experiments with different images.

## Command-line feature detection

### Corners
Usage: 
```sh
python corner-detector.py PATH_TO_IMAGE OUTPUT_NAME
```

## Command-line converter methods

### RGB to Local Entropy
Usage: 
```sh
python rgb2entropy.py PATH_TO_IMAGE PATH_TO_OUTPUT
```

### RGB to HSV Colormap
Usage: 
```sh
python rgb2hsv.py PATH_TO_IMAGE PATH_TO_OUTPUT
```

### RGB to Lab Colormap
Usage: 
```sh
python rgb2lab.py PATH_TO_IMAGE PATH_TO_OUTPUT
```

## Command-line segmentation methods

### K-Means
Usage: 
```sh
python3 kmeans.py PATH_TO_IMAGE N_CLUSTERS
```
Example: 
```sh
python3 kmeans.py PATH_TO_IMAGE 8
```

### Superpixels
Usage: 
```sh
python3 superpixel.py PATH_TO_IMAGE METHOD [PARAMS]
```

#### SLIC
Usage: 
```sh
python3 superpixel.py PATH_TO_IMAGE SLIC N_SEGMENTS SIGMA COMPACT
```

Example: 
```sh
python3 superpixel.py PATH_TO_IMAGE SLIC 400 1.0 30```

#### Felzenszwalb
Usage: 
```sh
python3 superpixel.py PATH_TO_IMAGE FELZENSZWALB SCALE SIGMA MIN_SIZE
```
Example: 
```sh
python3 superpixel.py PATH_TO_IMAGE FELZENSZWALB 100 0.5 50
```

#### Quickshift
Usage: 
```sh
python3 superpixel.py PATH_TO_IMAGE QUICKSHIFT KERNEL MAX_DIST RATIO
```
Example: 
```sh
python3 superpixel.py PATH_TO_IMAGE QUICKSHIFT 3 6 0.5
```

#### Watershed
Usage: 
```sh
python3 superpixel.py PATH_TO_IMAGE WATERSHED MARKERS COMPACT
```
Example: 
```sh
python3 superpixel.py PATH_TO_IMAGE WATERSHED 250 0.001
```

### Mean-Shift
Usage: 
```sh
python3 meanshift.py PATH_TO_IMAGE SCALE_FACTOR HS HR
```
Example: 
```sh
python3 meanshift.py PATH_TO_IMAGE 0.08 8 8
```

#### Watershed
Usage: 
```sh
python3 superpixel.py PATH_TO_IMAGE WATERSHED MARKERS COMPACT
```
Example: 
```sh
python3 superpixel.py PATH_TO_IMAGE WATERSHED 250 0.001
```

## Command-line Graph-cut methods

### Region Adjacency Graph [RAG](https://vcansimplify.wordpress.com/2014/07/06/scikit-image-rag-introduction/)
Usage: 
```sh
python3 graphcuts.py PATH_TO_IMAGE METHOD [PARAMS] CUTS_THRESHOLD
```
Example: 
```sh
python3 graphcut.py PATH_TO_IMAGE SLIC 400 1.0 30 30
```