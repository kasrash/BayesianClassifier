from src.dataset_gen import dataset_gen
from src.plot_funcs import sample_plot_3d, sample_plot_2d, db_plot, conf_matrix_plot
from src.discriminants import euclidean_distance
from src.classifier import classifier
from src.errors import error_calc, bhattacharyya_bound

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

## Settings
use_ds_p1 = True #swtich to use dataset from part 1; if true, must define the data path below
data_path = "C:\\Users\KasraShamsaei\Desktop\PR HW#1\Datasets\A\datasetA.csv"
plot_samples = True  # plot samples histogram in 3D; if use_ds_p1=False
plot_path = "C:\\Users\KasraShamsaei\Desktop\PR HW#1\plots\p3"

#Dataset A
mean_1 = np.array([1,1]) #mean
cov_1 = np.array([[1,0], [0,1]]) #covariance matrix
no_sample1 = 60000 #number of samples

mean_2 = np.array([4,4]) #mean
cov_2 = np.array([[1,0], [0,1]]) #covariance matrix
no_sample2 = 140000 #number of samples

if use_ds_p1:
    A = np.loadtxt(data_path, delimiter=',')
else:      
    A_1 = dataset_gen(mean_1, cov_1, no_sample1) #samples from dist. 1 with shape (samples, features)
    A_2 = dataset_gen(mean_2, cov_2, no_sample2) #samples from dist. 2 with shape (samples, features)
    A = np.vstack((A_1,A_2)) #generate dataset A
    if plot_samples:
        sample_plot_3d(A_1, 50, 1, plot_path)
        sample_plot_3d(A_2, 50, 2, plot_path)
        sample_plot_2d(A_1, 50, 3, plot_path)
        sample_plot_2d(A_2, 50, 4, plot_path)
    
true_classes = np.concatenate((np.ones(no_sample1,dtype=np.int8),2*np.ones(no_sample2,dtype=np.int8)),axis=0) #create true classes array for error calcs

# set prior probability of each class
p_omega1 = no_sample1 / (no_sample1+no_sample2)
p_omega2 = no_sample2 / (no_sample1+no_sample2)

# find the discriminant function values
g1 = euclidean_distance(A, mean_1, cov_1)
g2 = euclidean_distance(A, mean_2, cov_2)

# classification
pred_classes = classifier(g1, g2)

#error calculation
class_wise_counts, total_missed = error_calc(true_classes, pred_classes)
conf_matrix_plot(class_wise_counts, 6, plot_path)
conf_matrix_plot(class_wise_counts, 7, plot_path, normalize=True)
# bhattacharyya bound
p_error_battach = bhattacharyya_bound(mean_1, mean_2, cov_1, cov_2, p_omega1, p_omega2)

print("=========================================")
print("Classification Results of Part 3:")
print(f"Misclassification for class 1 is: {class_wise_counts[0,1]/(no_sample1)*100:0.2f} %")
print(f"Misclassification for class 2 is: {class_wise_counts[1,0]/(no_sample2)*100:0.2f} %")
print(f"Total misclassification is: {total_missed/(no_sample1+no_sample2)*100:0.2f} %")
print(f"\nBhattacharyya error bound is: {p_error_battach*100:0.2f} %")
print("=========================================")

