import cv2, os, sys
import matplotlib.pyplot as plt
import numpy as np
import itertools

from random import randint
from os.path import basename
from statsmodels.stats.inter_rater import cohens_kappa, to_table

inputs = sys.argv[1]
dataset_name = sys.argv[2]

precision_aux = []
recall_aux = []

tp_per_image = []
tn_per_image = []
fp_per_image = []
fn_per_image = []
accuracy_per_image = []
precision_per_image = []        
recall_per_image = []
f1_per_image = []

classes_names = ['background', 'roof', 'sky', 'wall', 'balcony', 'window', 'door', 'shop']
classes_colors =  [[0, 0, 0], #background
                  [0, 0, 255], #roof
                  [128, 255, 255], #sky
                  [255, 255, 0], #wall
                  [128, 0, 255], #balcony
                  [255, 0, 0], #window
                  [255, 128, 0], #door
                  [0, 255, 0]] #shop


def plot_confusion_matrix(cm, classes, normalize, cmap):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
   
    plt.savefig(dataset_name + ".pdf")
    

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


def checkClass(color):
    index = 0
    for cl_color in classes_colors:
        if(color == cl_color):
            return index
        else:
            index += 1

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
            

f = open(inputs)
lines = f.readlines()
cm_array = []
cm_sum = np.zeros((len(classes_colors), len(classes_colors)), dtype=int)
cm_aux = np.zeros((len(classes_colors), len(classes_colors)), dtype=int)

for line in lines:
    splited_line = line.split()
    annotation = splited_line[0]
    inference = splited_line[1]

    print("Validating image " + annotation + " with inference: " + inference)

    if((os.path.isfile(inference)) and (os.path.isfile(annotation))):    
        image = cv2.imread(inference)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = normalizeImage(image)

        gt = cv2.imread(annotation)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        gt = normalizeImage(gt)

        cm = np.zeros((len(classes_colors), len(classes_colors)), dtype=int)

        rows_image, cols_image, bands_image = image.shape
        im_matrix = np.zeros((rows_image, cols_image, bands_image), dtype=np.uint8)

        rows_gt, cols_gt, bands_gt = gt.shape
        im_gt = np.zeros((rows_gt, cols_gt, bands_gt), dtype=np.uint8)

        if((rows_image != rows_gt) or (cols_image != cols_gt)):
            print("Different dimension between original and reference images: " + inference)
            continue            

        total_pixels = rows_image * cols_image
        color_image_index = 0
        color_gt_index = 0
        n = 50000

        # randomly sweep the images
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

                if((color_image_index!=None) and (color_gt_index!=None)):
                    if(color_image == color_gt):                    
                        cm[color_image_index, color_image_index] += 1
                    else:
                        cm[color_gt_index, color_image_index] += 1
        else:
            print("N is bigger than number of pixels!")
        
        tp_array = []
        tn_array = []
        fp_array = []
        fn_array = []
        accuracy_array = []
        precision_array = []        
        recall_array = []
        f1_array = []
        for c in range(len(classes_names)): 
            tp = 0.0
            tn = 0.0
            fp = 0.0
            fn = 0.0
            accuracy = 0.0
            precision = 0.0
            recall = 0.0
            f1 = 0.0

            if(cm[c,c] == 0 or cm[c,c]=='nan'):
                tp_array.append(0)
                tn_array.append(0)
                fp_array.append(0)
                fn_array.append(0)
                continue
            else:
                tp = cm[c,c]
                tp_array.append(tp)

                fp = sumColumn(cm, c) - tp
                fp_array.append(fp)

                fn = sumRow(cm, c) - tp
                fn_array.append(fn)

                tn = cm.sum() - tp - fp - fn
                tn_array.append(tn)

                #accuracy = (tp + tn) / (tp + tn + fp + fn)
                #accuracy = cm.diagonal().sum() / float(n)
                accuracy = (tp + tn) / float(n)
                accuracy_array.append(accuracy)

                # sensitivity, recall, hit rate, or true positive rate (TPR)                                
                recall = tp / float(tp + fn)
                recall_array.append(recall)
                
                # specificity or true negative rate (TNR)
                #specificity = tn / (tn + fp)

                #precision or positive predictive value (PPV)       
                #precision_aux = np.true_divide(diagonal_fp, (diagonal_fp + fp)) 
                precision = tp / float(tp + fp)
                precision_array.append(precision)

                #prevalence: how often does the yes condition actually occur in our sample?
                #prevalence = (fn + tp) / (tp + tn + fp + fn)

                f1_score = 2 * ((precision * recall) / (precision + recall))
                f1_array.append(f1_score)
        
        #print(cohens_kappa(cm))

        # for each image, store confusion-matrix and the matrix for each classes
        cm_array.append(cm)
        tp_per_image.append(tp_array)
        tn_per_image.append(tn_array)
        fp_per_image.append(fp_array)
        fn_per_image.append(fn_array)
        accuracy_per_image.append(np.mean(accuracy_array))
        recall_per_image.append(np.mean(recall_array))
        precision_per_image.append(np.mean(precision_array))
        f1_per_image.append(np.mean(f1_array))

    else:
        print("Original or reference image is not a valid file. Check it and try again!")
        continue

sum_cm = 0
for i in range(len(classes_colors)):
    for j in range(len(classes_colors)):
        for k in range(len(cm_array)):
            cm_aux = cm_array[k]
            sum_cm = sum_cm + cm_aux[i,j]

        if(np.isnan(sum_cm)):
            sum_cm = 0

        cm_sum[i,j] = sum_cm        
        sum_cm = 0


tp_array = []
tn_array = []
fp_array = []
fn_array = []
for i in range(len(classes_colors)):
    count_tp = 0
    count_tn = 0
    count_fp = 0
    count_fn = 0
    for image in range(len(tp_per_image)):        
        count_tp += tp_per_image[image][i]
        count_tn += tn_per_image[image][i]
        count_fp += fp_per_image[image][i]
        count_fn += fn_per_image[image][i]

    tp_array.append(np.mean(count_tp))
    tn_array.append(np.mean(count_tn))
    fp_array.append(np.mean(count_fp))
    fn_array.append(np.mean(count_fn))


plot_confusion_matrix(cm_sum, classes=classes_names, normalize=True, cmap=plt.cm.Blues)
# cm_sum.print_stats()
# plt.show()

avg = np.mean(accuracy_per_image)
std = np.std(accuracy_per_image)
var = np.var(accuracy_per_image)

#avg_sen = np.mean(sensitivity_array)
#std_sen = np.std(sensitivity_array)
#var_sen = np.var(sensitivity_array)

avg_pr = np.mean(precision_per_image)
std_pr = np.std(precision_per_image)
var_pr = np.var(precision_per_image)

avg_rc = np.mean(recall_per_image)
std_rc = np.std(recall_per_image)
var_rc = np.var(recall_per_image)

avg_f1 = np.mean(f1_per_image)
std_f1 = np.std(f1_per_image)
var_f1 = np.var(f1_per_image)

print("tp: " + str(tp_array))
print("tn: " + str(tn_array))
print("fp: " + str(fp_array))
print("fn: " + str(fn_array))
print("Average accuracy: " + str(avg))
# print("Average sensitivity: " + str(avg_sen))
print("Average precision: " + str(avg_pr))
print("Average recall: " + str(avg_rc))
print("Average f1: " + str(avg_f1))

print("Variance accuracy: " + str(var))
#print("Variance sensitivity: " + str(var_sen))
print("Variance precision: " + str(var_pr))
print("Variance recall: " + str(var_rc))
print("Variance f1: " + str(var_f1))

print("Std accuracy: " + str(std))
# print("Std sensitivity: "+ str(std_sen))
print("Std precision: " + str(std_pr))
print("Std recall: " + str(std_rc))
print("Std f1: " + str(std_f1))

print("\n")