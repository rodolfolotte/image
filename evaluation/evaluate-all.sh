# echo 'RueMonge2014'
# python confusion-matrix.py /data/phd/results/facades-benchmark/ruemonge2014/inferences/50k/segmentation/IMG_5517.png /data/phd/data/facades-benchmark/ruemonge2014/annotation/IMG_5517.png
# python confusion-matrix.py /data/phd/results/facades-benchmark/ruemonge2014/inferences/50k/segmentation/IMG_5840.png /data/phd/data/facades-benchmark/ruemonge2014/annotation/IMG_5840.png
# python confusion-matrix.py /data/phd/results/facades-benchmark/ruemonge2014/inferences/50k/segmentation/IMG_5726.png /data/phd/data/facades-benchmark/ruemonge2014/annotation/IMG_5726.png

# echo 'CMP'
# python confusion-matrix.py /data/phd/results/facades-benchmark/cmp/inferences/50k/cmp_b0229.png /data/phd/data/facades-benchmark/cmp/annotation-unified/cmp_b0229.png
# python confusion-matrix.py /data/phd/results/facades-benchmark/cmp/inferences/50k/cmp_b0267.png /data/phd/data/facades-benchmark/cmp/annotation-unified/cmp_b0267.png
# python confusion-matrix.py /data/phd/results/facades-benchmark/cmp/inferences/50k/cmp_b0137.png /data/phd/data/facades-benchmark/cmp/annotation-unified/cmp_b0137.png

# echo 'eTRIMS'
# python confusion-matrix.py /data/phd/results/facades-benchmark/etrims/inferences/50k/basel_000062_mv0.png /data/phd/data/facades-benchmark/etrims/annotation-unified/basel_000062_mv0.png
# python confusion-matrix.py /data/phd/results/facades-benchmark/etrims/inferences/50k/heidelberg_000035_mv0.png /data/phd/data/facades-benchmark/etrims/annotation-unified/heidelberg_000035_mv0.png
# python confusion-matrix.py /data/phd/results/facades-benchmark/etrims/inferences/50k/bonn_000018.png /data/phd/data/facades-benchmark/etrims/annotation-unified/bonn_000018.png

# echo 'ECP'
# python confusion-matrix.py /data/phd/results/facades-benchmark/ecp/inferences/50k/monge_16.png /data/phd/data/facades-benchmark/ecp/annotation/monge_16.png
# python confusion-matrix.py /data/phd/results/facades-benchmark/ecp/inferences/50k/monge_80.png /data/phd/data/facades-benchmark/ecp/annotation/monge_80.png
# python confusion-matrix.py /data/phd/results/facades-benchmark/ecp/inferences/50k/monge_74.png /data/phd/data/facades-benchmark/ecp/annotation/monge_74.png

# echo 'ENPC'
# python confusion-matrix.py /data/phd/results/facades-benchmark/enpc/inferences/50k/facade_28.png /data/phd/data/facades-benchmark/enpc/annotation/facade_28.png
# python confusion-matrix.py /data/phd/results/facades-benchmark/enpc/inferences/50k/facade_73.png /data/phd/data/facades-benchmark/enpc/annotation/facade_73.png
# python confusion-matrix.py /data/phd/results/facades-benchmark/enpc/inferences/50k/facade_54.png /data/phd/data/facades-benchmark/enpc/annotation/facade_54.png

# echo 'Graz'
# python confusion-matrix.py /data/phd/results/facades-benchmark/graz/inferences/50k/facade_0_0053679_0053953.png /data/phd/data/facades-benchmark/graz/labels_full/facade_0_0053679_0053953.png
# python confusion-matrix.py /data/phd/results/facades-benchmark/graz/inferences/50k/facade_1_0056092_0056345.png /data/phd/data/facades-benchmark/graz/labels_full/facade_1_0056092_0056345.png
# python confusion-matrix.py /data/phd/results/facades-benchmark/graz/inferences/50k/facade_0_0100003_0100182.png /data/phd/data/facades-benchmark/graz/labels_full/facade_0_0100003_0100182.png

echo 'SJC-Ruemonge2014'
python confusion-matrix.py /data/phd/results/facades-benchmark/sjc/inferences/ruemonge-knowledge/segmentation-sjc/IMG_7754.png /data/phd/data/facades-benchmark/sjc/annotation/IMG_7754.png

echo 'SJC-CMP'
python confusion-matrix.py /data/phd/results/facades-benchmark/sjc/inferences/cmp-knowledge/segmentation-sjc/IMG_7754.png /data/phd/data/facades-benchmark/sjc/annotation/IMG_7754.png

echo 'SJC-eTRIMS'
python confusion-matrix.py /data/phd/results/facades-benchmark/sjc/inferences/etrims-knowledge/segmentation-sjc/IMG_7754.png /data/phd/data/facades-benchmark/sjc/annotation/IMG_7754.png

echo 'SJC-ECP'
python confusion-matrix.py /data/phd/results/facades-benchmark/sjc/inferences/ecp-knowledge/segmentation-sjc/IMG_7754.png /data/phd/data/facades-benchmark/sjc/annotation/IMG_7754.png

echo 'SJC-ENPC'
python confusion-matrix.py /data/phd/results/facades-benchmark/sjc/inferences/enpc-knowledge/segmentation-sjc/IMG_7754.png /data/phd/data/facades-benchmark/sjc/annotation/IMG_7754.png

echo 'SJC-Graz'
python confusion-matrix.py /data/phd/results/facades-benchmark/sjc/inferences/graz-knowledge/segmentation-sjc/IMG_7754.png /data/phd/data/facades-benchmark/sjc/annotation/IMG_7754.png

echo 'All-together-SJC'
python confusion-matrix.py /data/phd/results/facades-benchmark/sjc/inferences/all-together-knowledge/segmentation-sjc/IMG_7754.png /data/phd/data/facades-benchmark/sjc/annotation/IMG_7754.png