import cv2, os, sys
import matplotlib.pyplot as plt
import numpy as np
import itertools
import time

from random import randint
from os.path import basename
from statsmodels.stats.inter_rater import cohens_kappa, to_table

original = sys.argv[1]
reference = sys.argv[2]

original_base = basename(original)
reference_base = basename(reference)

accuracy_array = []
sensitivity_aux = []
sensitivity_array = []
precision_aux = []
precision_array = []
recall_aux = []
recall_array = []
f1_array = []

classes_names = ['background', 'roof', 'sky', 'wall', 'balcony', 'window', 'door', 'shop']
classes_colors =  [[0, 0, 0], #background
                  [0, 0, 255], #roof
                  [128, 255, 255], #sky
                  [255, 255, 0], #wall
                  [128, 0, 255], #balcony
                  [255, 0, 0], #window
                  [255, 128, 0], #door
                  [0, 255, 0]] #shop

def plot_confusion_matrix(cm, classes, normalize, title, cmap):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Confusion matrix normalized")
    else:
        print('Confusion matrix')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def checkClass(color):
    index = 0
    for cl_color in classes_colors:
        if(color == cl_color):
            return index
        else:
            index += 1

def sumColumn(matrix, column):
    total = 0
    for row in range(len(matrix)):
        total += matrix[row][column]
    return total

def sumRow(matrix, row):
    total = 0
    for column in range(len(matrix)):
        total += matrix[row][column]
    return total

def checkZeroDiagonal(confusion_matrix_diag, array):
    indexes = []
    for i in range(len(confusion_matrix_diag)):         
        if(confusion_matrix_diag[i] == 0 and array[i] == 0):                        
            indexes.append(i)
    new_cm = np.delete(confusion_matrix_diag, indexes)
    new_array = np.delete(array, indexes)                                

    return new_cm, new_array


def normalizeImage(image):
    rows_image, cols_image, bands_image = image.shape
    color_new_image = []
    for i in range(rows_image):
        for j in range(cols_image):
            color_image = image[i, j]

            for b in range(bands_image):
                if (color_image[b] > 0 and color_image[b] <= 70):		
                    color_image[b] = 0
            
                if (color_image[b] > 70 and color_image[b] <= 140):		
                    color_image[b] = 128
                
                if (color_image[b] > 140 and color_image[b] < 256):		
                    color_image[b] = 255

            image[i, j] = color_image

    return image

    
if((os.path.isfile(original)) and (os.path.isfile(reference))):
    print("Validating image " + original_base + " with reference: " + reference_base)
    image = cv2.imread(original)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = normalizeImage(image)

    gt = cv2.imread(reference)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    gt = normalizeImage(gt)

    confusion_matrix = np.zeros((len(classes_colors), len(classes_colors)), dtype=int)

    rows_image, cols_image, bands_image = image.shape
    im_matrix = np.zeros((rows_image, cols_image, bands_image), dtype=np.uint8)

    rows_gt, cols_gt, bands_gt = gt.shape
    im_gt = np.zeros((rows_gt, cols_gt, bands_gt), dtype=np.uint8)

    if((rows_image != rows_gt) or (cols_image != cols_gt)):
        print("Different dimension between original and reference images: " + original_base)
        sys.exit()

    total_pixels = rows_image * cols_image
    color_image_index = 0
    color_gt_index = 0

    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    accuracy = 0.0
    sensitivity = 0.0
    specificity = 0.0
    precision = 0.0
    prevalence = 0.0
    f1_score = 0.0
    error_rate = 0.0

    n = 200000
    
    if(n < total_pixels):
        for k in range(n-1):
            i = randint(0, rows_image-1)
            j = randint(0, cols_image-1)

            color_image = image[i, j]
            color_image = map(int, color_image)
            color_image_index = checkClass(color_image)

            color_gt = gt[i, j]
            color_gt = map(int, color_gt)
            color_gt_index = checkClass(color_gt)

            if((color_image_index != None) and (color_gt_index != None)):
                if(color_image == color_gt):                                
                    confusion_matrix[color_image_index, color_image_index] += 1                
                else:                          
                    confusion_matrix[color_gt_index, color_image_index] += 1

            # print(str(color_image) + " is equal of " + str(color_gt) + ": " + str(color_image == color_gt) + " class_image: " + str(color_image_index) + " class_gt: " + str(color_gt_index))
                  column
    else:
        print("N is bigger than number of pixels!")

    tp_array = []
    tn_array = []
    fp_array = []
    fn_array = []
    accuracy_array = []
    recall_array = []
    precision_array = []
    f1_array = []
    for c in range(len(classes_names)):        
        if(confusion_matrix[c,c] == 0 or confusion_matrix[c,c]=='nan'):
            tp_array.append(0)
            tn_array.append(0)
            fp_array.append(0)
            fn_array.append(0)
            continue
        else:
            tp = confusion_matrix[c,c]
            tp_array.append(tp)

            tn = confusion_matrix.diagonal().sum() - tp
            tn_array.appned(tn)

            fp = sumColumn(confusion_matrix, c) - tp
            fp_array.append(fp)

            fn = sumRow(confusion_matrix, c) - tp
            fn_array.append(fn)

            accuracy = (tp + tn) / float(n)
            accuracy_array(accuracy)

            recall = tp / (tp + fn)
            recall_array(recall)

            precision = tp / (tp + fp)
            precision_array(precision)

            f1_score = 2 * ((precision * recall) / (precision + recall))
            f1_array(f1_score)
    
    print(tp_array)
    print(tn_array)
    print(fp_array)
    print(fn_array)
    print(accuracy_array)
    print(f1_array)



    # tp = confusion_matrix.diagonal().sum()
    # tn = confusion_matrix.diagonal().sum()
    # fp = np.sum(confusion_matrix, axis=0) - confusion_matrix.diagonal()
    # fn = np.sum(confusion_matrix, axis=1) - confusion_matrix.diagonal()

    # diagonal_fp, fp = checkZeroDiagonal(confusion_matrix.diagonal(), fp)
    # diagonal_fn, fn = checkZeroDiagonal(confusion_matrix.diagonal(), fn)

    # accuracy
    #accuracy = (tp + tn) / (tp + tn + fp + fn)
    #accuracy = (tp + tn) / float(n)

    # sensitivity, recall, hit rate, or true positive rate (TPR)
    # sensitivity = tp / (tp + fn)
    #sensitivity_aux = np.true_divide(diagonal_fn, (diagonal_fn + fn))
    #sensitivity = np.mean(sensitivity_aux)

    # specificity or true negative rate (TNR)
    #specificity = tn / (tn + fp)

    #precision or positive predictive value (PPV)
    #precision_aux = np.true_divide(diagonal_fp, (diagonal_fp + fp))
    #precision = np.mean(precision_aux)
    
    # prevalence: how often does the yes condition actually occur in our sample?
    #prevalence = (fn + tp) / (tp + tn + fp + fn)

    # F1 score is the harmonic mean of precision and recall
    # f1_score = (2 * tp) / (2*tp + fp + fn)
    #f1_score = 2 * ((precision * sensitivity) / (precision + sensitivity))
        
    # error rate or mis-classification rate: overall, how often is it wrong?
    #error_rate = 1 - accuracy

    #print("...confusion matrix:")
    #print(confusion_matrix)
    #plot_confusion_matrix(confusion_matrix, classes=classes_names, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues)
    #plt.show()

    #print("...total pixels: " + str(total_pixels))
    #print("...random samples: " + str(n))
    #print("...overall accuracy: " + '{0:.3f}'.format(accuracy))
    
    # print("...sensitivity/recall/hit rate/true positive rate: " + str(sensitivity))
    # print("...specificity/true negative rate: " + '{0:.3f}'.format(specificity))
    #print("...precision/positive predictive value: " + str(precision))
    # print("...prevalence: " + '{0:.3f}'.format(prevalence))
    # print("...f1-score: " + '{0:.3f}'.format(f1_score))
    # print("...error rate: " + '{0:.3f}'.format(error_rate))
    #print("...kappa coefficient: ")
    #print(cohens_kappa(confusion_matrix))


else:
    print("Original or reference image is not a valid file. Check it and try again!")
