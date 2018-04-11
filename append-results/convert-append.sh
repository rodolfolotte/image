#! /bin/bash

# Usage: ./convert-append.sh /home/rodolfo/Desktop/orig/ /home/rodolfo/Desktop/seg/ /home/rodolfo/Desktop/overlay/ /home/rodolfo/Desktop/merge1/
for file in $1*.JPG; do    
	noextension="${file%.*}"
    filename=$(basename "$noextension")
    segmentation=$2$filename'.png'

    overlay=$3$filename'.png'
    merge1=$4$filename'.png'
    #merge2=$5$filename'.png'

    if [ -f $segmentation ]; then
    	composite -blend 30 $segmentation $file $overlay
     	convert -append $file $segmentation $overlay $4$filename'.png'
	fi
done

#for file in $4*.png; do
#	convert +append *.png $5'result.png'
#done
