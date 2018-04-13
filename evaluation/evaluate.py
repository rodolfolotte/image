import cv2, os, sys
import matplotlib.pyplot as plt
import numpy as np

from os.path import basename

original = sys.argv[1]
reference = sys.argv[2]

original_base = basename(original)
reference_base = basename(reference)

classes_colors =  [[0, 0, 0], #background
                  [0, 0, 255], #roof
                  [128, 255, 255], #sky
                  [255, 255, 0], #wall
                  [128, 0, 255], #balcony
                  [255, 0, 0], #window
                  [255, 128, 0], #door
                  [0, 255, 0]] #shop

print(os.path.isfile(original))
print(os.path.isfile(reference))

if((os.path.isfile(original)) and (os.path.isfile(reference))):
    print("Validating image " + original_base + " with reference: " + reference_base)
    image = cv2.imread(original)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gt = cv2.imread(reference)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

    rows_image, cols_image, bands_image = image.shape
    im_matrix = np.zeros((rows_image, cols_image, bands_image), dtype=np.uint8)

    rows_gt, cols_gt, bands_gt = gt.shape
    im_gt = np.zeros((rows_gt, cols_gt, bands_gt), dtype=np.uint8)

    if((rows_image != rows_gt) or (cols_image != cols_gt)):
        print("Different dimension between original and reference images: " + original_base)
        sys.exit()

    total_pixels = rows_image * cols_image
    
    count_colors_in_image_array = []
    count_colors_in_gt_array = []
    tp = []
    tn = []
    fp = []
    fn = []
    accuracy_array = []
    precision_array = []

    count_colors_in_image = 0
    count_colors_in_gt = 0
    tp_count = 0
    tn_count = 0
    fp_count = 0
    fn_count = 0
    accuracy = 0.0
    precision = 0.0

    count_real_classes = 0
    
    #True positive = correctly identified
    #False positive = incorrectly identified
    #True negative = correctly rejected
    #False negative = incorrectly rejected
    for cl_color in classes_colors:
        for i in range(rows_image):
            for j in range(cols_image):                

                color_image = image[i, j]
                color_image = map(int, color_image)                

                color_gt = gt[i, j]
                color_gt = map(int, color_gt)
                
                if(cl_color == color_image):
                    count_colors_in_image += 1

                if(cl_color == color_gt):
                    count_colors_in_gt += 1

                if((color_gt == cl_color) and (color_image == color_gt)):
                    tp_count += 1
                
                if((color_gt != cl_color) and (color_image != color_gt)):
                    tn_count += 1

                if((color_gt != cl_color) and (color_image == cl_color)):
                    fp_count += 1
                
                if((color_gt == cl_color) and (color_image != cl_color)):
                    fn_count += 1

        if(count_colors_in_image == 0):
            tp.append(count_colors_in_image)
            fp.append(count_colors_in_image)
            tn.append(count_colors_in_image)
            fn.append(count_colors_in_image)
        else:
            count_real_classes += 1
            count_colors_in_image_array.append(count_colors_in_image)
            count_colors_in_gt_array.append(count_colors_in_gt)

            if(count_colors_in_image != 0):
                tp.append(tp_count/float(count_colors_in_image))
                fp.append(fp_count/float(count_colors_in_image))
                tn.append(tn_count/float(count_colors_in_image))
                fn.append(fn_count/float(count_colors_in_image))
            else:
                tp.append(tp_count)
                fp.append(fp_count)
                tn.append(tn_count)
                fn.append(fn_count)
            
            acc_division = float(tp_count + fp_count + tn_count + fn_count)
            pre_division = float(tp_count + fp_count)

            if(acc_division != 0.0):
                accuracy_array.append((tp_count + fp_count) / acc_division)
            else:
                accuracy_array.append(0)

            if(pre_division != 0.0):
                precision_array.append(tp_count / pre_division)
            else:
                precision_array.append(0)

            print('....class: ' + str(cl_color) + ": tp[" + str(tp_count) + " - " + str(tp_count/float(count_colors_in_image)) + "] tn[" + str(tn_count) + " - " + str(tn_count/float(count_colors_in_image)) + "] fp[" + str(fp_count) + " - " + str(fp_count/float(count_colors_in_image)) + "] fn[" + str(fn_count) + " - " + str(fn_count/float(count_colors_in_image)) + "] - population; " + str(count_colors_in_image))

        count_colors_in_image = 0
        count_colors_in_gt = 0
        tp_count = 0
        tn_count = 0
        fp_count = 0
        fn_count = 0
    
    print("...total pixels: " + str(total_pixels))
    print("...expected classes: " + str(len(classes_colors)) + " found in segmented image: " + str(count_real_classes))
    sys.stdout.write("...accuracy: ")
    print(" ".join(str(x) for x in accuracy_array))    
    sys.stdout.write("...precision: ")
    print(" ".join(str(x) for x in precision_array))  

    overall_acc = sum(accuracy_array) / float(count_real_classes)
    overall_pre = sum(precision_array) / float(count_real_classes)      

    print("..overall accuracy: " + str(overall_acc))    
    print("..overall precision: " + str(overall_pre) + "\n")

else:
    print("Original or reference image is not a valid file. Check it and try again!")
