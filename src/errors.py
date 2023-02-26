import numpy as np

def error_calc(true_classes, pred_classes):
    '''
    Calculate the class-wise correct classifications and misclassification as well as total misclassifications.

    Inputs:
    true_classes = ground truth
    pred_classes = predicted classes

    Return:
    class_wise_counts = confusion matrix of the classification
    total_missed = total number of misclassified labels
    '''
    count11 = 0; count12 = 0; count21 = 0; count22 = 0;  #count the correct classification and misclassifications; countij = number of samples with true class i that is classified as class j
    for ss in range(true_classes.shape[0]):
        if pred_classes[ss] == true_classes[ss]:
           if pred_classes[ss] == 1:
               count11 += 1
           else:
               count22 += 1
        else: #since we have only 2 classes
            if pred_classes[ss] == 1: #if predicted as 1 but it was 2
                count21 += 1
            else:
                count12 += 1
     
    class_wise_counts = np.array([[count11, count12],[count21, count22]])
    total_missed = np.sum(np.where(pred_classes==true_classes, 0, 1))

    return class_wise_counts, total_missed

def bhattacharyya_bound(mean_1, mean_2, cov_1, cov_2, p_omega1, p_omega2):
    '''
    Calculate the bhattacharyya error bound.

    Inputs:
    mean_1 = mean of class 1
    mean_2 = mean of class 2
    cov_1 = covariance matrix of class 1
    cov_2 = covariance matrix of class 2
    p_omega1 = prior probability of class 1
    p_omega2 = prior probability of class 2

    Return:
    p_error = bhattacharyya error bound
    '''
    beta = 0.5

    k_beta = 0.5*beta*(1-beta)*np.matmul(np.matmul(mean_1-mean_2,np.linalg.inv((1-beta)*cov_1+beta*cov_2)),mean_1-mean_2) + 0.5*np.log(np.linalg.det((1-beta)*cov_1+beta*cov_2)/(np.linalg.det(cov_1)**(1-beta)*np.linalg.det(cov_2)**beta))

    p_error = p_omega1**beta * p_omega2**(1-beta) * np.exp(-1.*k_beta) # refer to report for the equations

    return p_error