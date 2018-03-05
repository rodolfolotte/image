import urllib
import os
import numpy
#import pygmaps
import webbrowser
import requests
import json
from pprint import pprint

lotte_gsv_images_path = "/home/lotte/Desktop/street-view-images/"
lotte_key = "AIzaSyBgznlJ3ZLDMAKGIf9Exx2B3thSA6t32ZQ"

SV_IMAGE = "https://maps.googleapis.com/maps/api/streetview?"
SV_IMAGE_METADATA = "https://maps.googleapis.com/maps/api/streetview/metadata?"

# get a list of streets based only the name of the city or area
#streets = ["Topaasstraat, Amsterdam, North Holand"]
streets = ["Avenida 9 de Julho, Sao Jose dos Campos, Brasil"]
WIDTH = 640
HEIGHT = 640
HEADING_MIN=0
HEADING_MAX=360
HEADING_STEP=45
PITCH_MIN=15
PITCH_MAX=45
PITCH_STEP=15

# TODO
def getStreetSpots(StreetLocation):
  latlon = [(52.350872, 4.907210), (52.350644, 4.907293), (52.350444, 4.907365), (52.350241, 4.907444), (52.350028, 4.907511), (52.349802, 4.907614)]
  return latlon


def getImage(StreetSpots, Heading, Pitch, SaveLoc):
  size = "size=" + str(WIDTH) + "x" + str(HEIGHT)
  heading = "heading=" + str(Heading)
  pitch = "pitch=" + str(Pitch)
  location = "location=" + str(StreetSpots[0]) + ',' + str(StreetSpots[1])
  key = "key=" + lotte_key

  url = SV_IMAGE + size + "&" + heading + "&" + pitch + "&" + location + "&" + key

  fi = str(StreetSpots[0]) + "_" + str(StreetSpots[1]) + "_" + str(Heading) + "_" + str(Pitch) + ".jpg"
  #fi = "_" + str(Heading) + "_" + str(Pitch) + ".jpg"
  r = urllib.urlretrieve(url, os.path.join(SaveLoc,fi))

if __name__ == '__main__':
  for s in streets:
    street_spots = getStreetSpots(StreetLocation=s)
    for sp in street_spots:
      for h in range(HEADING_MIN,HEADING_MAX,HEADING_STEP): # nao ha necessidade de baixar o heading 180 - ao longo da rua
        for p in range(PITCH_MIN,PITCH_MAX,PITCH_STEP):
          getImage(StreetSpots=sp, Heading=h, Pitch=p, SaveLoc=lotte_gsv_images_path)

#TODO
# plota os pins de todos os pontos
#mymap.addpoint(37.427, -122.145, "#0000FF")
#path = [(37.429, -122.145),(37.428, -122.145),(37.427, -122.145),(37.427, -122.146),(37.427, -122.146)]

#mymap.draw('./mymap.draw.html')
#url = './mymap.draw.html'
#webbrowser.open_new_tab(url)