from src.BoxMuller import box_muller

import numpy as np

def dataset_gen(mean, cov, no_samples):
    '''
    Generates samples using Box-Muller transformation for a given 2D Gaussian dist. mean and covariance matrix as well as number of required samples.

    Inputs:
    mean = mean of the Gaussian dist.
    cov = covariance matrix of the Gaussian dist.
    no_samples = number of samples to generate

    Returns:
    samples = samples drawn from the given 2D Gaussian dist. with shape (samples, features)
    '''
    x1 = np.expand_dims(box_muller(mean[0], cov[0,0], no_samples),axis=1) # make it 2D to be able to concatenate
    x2 = np.expand_dims(box_muller(mean[1], cov[1,1], no_samples),axis=1) # make it 2D to be able to concatenate
    samples = np.concatenate((x1,x2),axis=1)  

    return samples