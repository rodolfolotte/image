#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

"""
Poligonizer - Transform imagens from raster to shapefiles
"""

__author__ = 'Rodolfo G. Lotte'
__copyright__ = 'Copyright 2018, Rodolfo G. Lotte'
__credits__ = ['Rodolfo G. Lotte']
__license__ = 'MIT'
__email__ = 'rodolfo.lotte@gmail.com'

import os
import gdal
import osgeo.osr as osr, ogr
import logging
import cv2
import datetime
import json

from os.path import basename

classes = {
    "deforestation": [255,255,0]
  }

def createGeometries(corners, hierarchy, image):
    gt = image.GetGeoTransform()
    poly = ogr.Geometry(ogr.wkbPolygon)
    geom = []

    for i in range(len(corners)):
        flag = True
        ring = ogr.Geometry(ogr.wkbLinearRing)

        for coord in corners[i]:
            Xgeo = gt[0] + coord[0][0] * gt[1] + coord[0][1] * gt[2]
            Ygeo = gt[3] + coord[0][0] * gt[4] + coord[0][1] * gt[5]
            ring.AddPoint(Xgeo, Ygeo)

            if (flag == True):
                flag = False
                initialX = Xgeo
                initialY = Ygeo

        ring.AddPoint(initialX, initialY)
        poly.AddGeometry(ring)

        # TODO: refatorar condicionais
        if(i+1 < len(corners)):
            if((hierarchy[0, i, 2] != -1) or ((hierarchy[0, i, 2] == -1) and (hierarchy[0, i, 3] != -1) and (hierarchy[0, i+1, 3] != -1))):
                continue
            else:
                geom.append(ogr.CreateGeometryFromWkt(poly.ExportToWkt()))
                poly = ogr.Geometry(ogr.wkbPolygon)
        else:
                geom.append(ogr.CreateGeometryFromWkt(poly.ExportToWkt()))
                poly = ogr.Geometry(ogr.wkbPolygon)

    return geom


def getClassesGT(complete_path_png, hypes):
    imageSegmented = cv2.imread(complete_path_png)
    imageSegmented = cv2.cvtColor(imageSegmented, cv2.COLOR_BGR2RGB)

    height, width, bands = imageSegmented.shape

    gt_classes = []
    for key, value in hypes['which_segments'].items():
        for i in range(height):
            for j in range(width):
                if ((imageSegmented[i, j][0] == value[0]) and (imageSegmented[i, j][1] == value[1]) and (imageSegmented[i, j][2] == value[2]) and (key not in gt_classes)):
                    gt_classes.append(key)

    return gt_classes


def getImageByClass(complete_path_png, key, classes):
    imageSegmented = cv2.imread(complete_path_png)
    imageSegmented = cv2.cvtColor(imageSegmented, cv2.COLOR_BGR2RGB)

    value = classes[key]
    height, width, bands = imageSegmented.shape

    for i in range(height):
        for j in range(width):
            if not ((imageSegmented[i, j][0] == value[0]) and (imageSegmented[i, j][1] == value[1]) and (imageSegmented[i, j][2] == value[2])):
                imageSegmented[i, j][0] = 0
                imageSegmented[i, j][1] = 0
                imageSegmented[i, j][2] = 0

    return imageSegmented



def createShapefile(segmented, complete_path_metadata, complete_path_vector):
    filename = basename(complete_path_png)
    name = os.path.splitext(filename)[0].encode('utf-8')

    image = gdal.Open(complete_path_metadata)

    #driver = ogr.GetDriverByName('ESRI Shapefile')
    driver = ogr.GetDriverByName('GeoJSON')

    if (os.path.isfile(complete_path_vector)):
        os.remove(complete_path_vector)

    ds = driver.CreateDataSource(complete_path_vector)
    if(ds==None):
        logging.info(">>>> Problems with vector file {}. Polygonization interrupted!".format(complete_path_vector))
        return

    srs = osr.SpatialReference()
    srs.ImportFromWkt(image.GetProjection())

    _area_pix = ogr.FieldDefn('area-pix', ogr.OFTReal)
    _class = ogr.FieldDefn('class', ogr.OFTString)

    gt_classes = getClassesGT(segmented, hypes)

    classesAndGeometries = {}
    for k in range(len(gt_classes)):
        imageSegmented = getImageByClass(segmented, gt_classes[k], classes)
        imageSegmentedInGray = cv2.cvtColor(imageSegmented, cv2.COLOR_RGB2GRAY)

        thresh = cv2.threshold(imageSegmentedInGray, 127, 255, 0)[1]
        im2, corners, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        geometries = createGeometries(corners, hierarchy, image)

        classesAndGeometries[gt_classes[k]] = geometries

    layer = ds.CreateLayer(name, srs, ogr.wkbPolygon)

    layer.CreateField(_area_pix)
    layer.CreateField(_class)

    for key, value in classesAndGeometries.items():
        for g in range(len(value)):
            featureDefn = layer.GetLayerDefn()
            feature = ogr.Feature(featureDefn)

            # the area of the feature in square units of the spatial reference system in use
            area = value[g].GetArea()
            if(area == 0): continue

            feature.SetGeometry(value[g])
            feature.SetField('area-pix', float(area))
            feature.SetField('class', str(key))
	        #more attributes could be inserted here

            layer.CreateFeature(feature)

    logging.info(">>>> Vector file of image {} created!".format(filename))


def polygonize(segmented, image, output):
    valid_images = [".png"]

    segmented_filename = os.path.basename(segmented)
    polygon_name = segmented_filename.split('.')[0] + '.geojson'
    full_out_polygon = os.path.join(output, polygon_name)


    if ext.lower() not in valid_images:
        logging.info(">>>> Image with no accept extention: " + str(ext) + ".")

    if((not os.path.isfile(json)) or (not os.path.isfile(json))):
        logging.info(">>>> Image or JSON is not present for file: " + image + ".")
        return

    if ((not os.path.isfile(segmented))):
        logging.info(">>>> Image PNG is not present for file: " + image + ".")
        return

    createShapefile(segmented, image, full_out_polygon)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Poligonizer - Transform imagens from raster to shapefiles')
    
    parser.add_argument('-image', action="store", dest='imageSegmented', help='Image segmented, or image to be vectorized')
    parser.add_argument('-image_with_metadata', action="store", dest='imageMetadata', help='Original image, which contains the spatial references')
    parser.add_argument('-output_folder', action="store", dest='outputFolder', help='Folder to store the ouptup files')

    result = parser.parse_args()

    logging.info("Polygonizing inputs...")
    polygonize(result.imageSegmented, result.imageMetadata, result.outputFolder)
