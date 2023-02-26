import numpy as np

def case1_hp(mean_1, mean_2, cov, p_omega1, p_omega2, bounds=[[-5,10],[-5,10]], delta=0.1):
    '''
    Generate the value of the function derived in the report to find the decision boundary for case 1.

    Input:
    mean_1 = mean of dist. 1
    mean_2 = mean of dist. 2
    cov = covariance matrix (since cov_1==cov_2 in case 1, just one cov needs to be passed)
    p_omega1 = prior probability of class 1
    p_omega2 = prior probability of class 2
    bounds = bounds to generate mesh with shape [[x1_s, x1_e], [x2_s,x2_e]]
    delta = mesh size

    Return:
    hp = the values of the function to generate decision boundary from
    x1 = mesh values for plotting in x1 direction
    x2 = mesh values for plotting in x2 direction
    '''
    std = cov[0,0]
    # refer to the report for these calculations
    w = mean_1 - mean_2
    x0 = 0.5*(mean_1+mean_2) - (std/np.matmul(mean_1-mean_2,mean_1-mean_2))*np.log(p_omega1/p_omega2)*(mean_1-mean_2) #mohammad can you please chekc this?
    # generate mesh
    x1 = np.arange(bounds[0][0],bounds[0][1]+delta,delta)
    x2 = np.arange(bounds[1][0],bounds[1][1]+delta,delta)
    # generate values of the function to find decision boundary from
    hp = np.zeros((x1.shape[0],x2.shape[0]))
    for x1id in range(x1.shape[0]):
        for x2id in range(x2.shape[0]):
            x = np.array([x1[x1id],x2[x2id]])
            hp[x1id, x2id] = np.matmul(w, x-x0) # refer to the report for the equation of the function

    return hp, x1, x2