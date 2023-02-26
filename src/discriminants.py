import numpy as np

def case1_discriminant(dataset, mean, cov, p_omega):
    '''
    Generate the values of the discriminant function for Case 1.

    Input:
    dataset = dataset with shape (dataset, features)
    mean = mean of the dist for which to create the discriminant function values
    cov = covariance matrix of the dist for which to create the discriminant function values
    p_omega = prior probability of the target class for which to create the discriminant function values

    Return:
    g = discriminant function values for the given class
    '''
    std = cov[0,0]
    # calculate the discriminant function values
    g = []
    for ss in dataset:
        temp_g = -1.* (np.matmul(ss-mean,ss-mean)/std/2) + np.log(p_omega) # refer to the report for the equations
        g.append(temp_g)

    return np.array(g)

def euclidean_distance(dataset, mean, cov):
    '''
    Generate the values of the Euclidean distance classifier.

    Input:
    dataset = dataset with shape (dataset, features)
    mean = mean of the dist for which to create the discriminant function values
    cov = covariance matrix of the dist for which to create the discriminant function values

    Return:
    g = discriminant function values for the given class
    '''
    std = cov[0,0]
    # calculate the discriminant function values
    g = []
    for ss in dataset:
        temp_g = -1. * np.matmul(ss-mean,ss-mean) # refer to the report for the equations
        g.append(temp_g)

    return np.array(g)