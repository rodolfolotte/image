import cv2, os, sys
import matplotlib.pyplot as plt
import numpy as np
import itertools

from random import randint
from os.path import basename
from statsmodels.stats.inter_rater import cohens_kappa, to_table

inputs = sys.argv[1]
dataset_name = sys.argv[2]

accuracy_array = []
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
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
   
    plt.savefig(dataset_name + ".pdf")
    

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

        tp = 0.0
        tn = 0.0
        fp = 0.0
        fn = 0.0
        accuracy = 0.0
        sensitivity = 0.0
        specificity = 0.0
        precision = 0.0
        recall = 0.0
        prevalence = 0.0
        f1 = 0.0
        error_rate = 0.0

        n = 100000

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

        tp = cm.diagonal().sum()
        #tn = np.sum(np.triu(cm)) + np.sum(np.tril(cm))
        fp = np.sum(cm, axis=0) - cm.diagonal()
        fn = np.sum(cm, axis=1) - cm.diagonal()

        diagonal_fp, fp = checkZeroDiagonal(cm.diagonal(), fp)
        diagonal_fn, fn = checkZeroDiagonal(cm.diagonal(), fn)

        # accuracy
        #accuracy = (tp + tn) / (tp + tn + fp + fn)
        accuracy = cm.diagonal().sum() / float(n)
        accuracy_array.append(accuracy)

        # sensitivity, recall, hit rate, or true positive rate (TPR)
        #sensitivity = tp / (tp + fn)
        #sensitivity_array.append(sensitivity)  
        
        # specificity or true negative rate (TNR)
        #specificity = tn / (tn + fp)

        #precision or positive predictive value (PPV)       
        precision_aux = np.true_divide(diagonal_fp, (diagonal_fp + fp))        
        precision = np.mean(precision_aux)        
        precision_array.append(precision)

        # recall
        recall_aux = np.true_divide(diagonal_fn, (diagonal_fn + fn))
        recall = np.mean(recall_aux)
        recall_array.append(recall)

        # F1 score is the harmonic mean of precision and sensitivity
        f1 = 2 * ((precision * recall) / (precision + recall))
        f1_array.append(f1)
                
        # prevalence: how often does the yes condition actually occur in our sample?
        #prevalence = (fn + tp) / (tp + tn + fp + fn)

        #print(cohens_kappa(cm))

        cm_array.append(cm)

    else:
        print("Original or reference image is not a valid file. Check it and try again!")
        continue

sum_cm = 0
for i in range(len(classes_colors)):
    for j in range(len(classes_colors)):
        for k in range(len(cm_array)):
            cm_aux = cm_array[k]
            sum_cm = sum_cm + cm_aux[i,j]

        cm_sum[i,j] = sum_cm
        sum_cm = 0

plot_confusion_matrix(cm_sum, classes=classes_names, normalize=True, cmap=plt.cm.Blues)
# sum_cm.print_stats()
# plt.show()

avg = np.mean(accuracy_array)
std = np.std(accuracy_array)
var = np.var(accuracy_array)

#avg_sen = np.mean(sensitivity_array)
#std_sen = np.std(sensitivity_array)
#var_sen = np.var(sensitivity_array)

avg_pr = np.mean(precision_array)
std_pr = np.std(precision_array)
var_pr = np.var(precision_array)

avg_rc = np.mean(recall_array)
std_rc = np.std(recall_array)
var_rc = np.var(recall_array)

avg_f1 = np.mean(f1_array)
std_f1 = np.std(f1_array)
var_f1 = np.var(f1_array)

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