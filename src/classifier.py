import numpy as np

def classifier(g1, g2):
    '''
    Classify the samples based on the given discriminant functions values.

    Inputs:
    g1 = discriminant functions values based on class 1
    g2 = discriminant functions values based on class 2

    Return:
    pred_classes = array of predicted classes using the discriminant functions values
    '''
    steps = g1.shape[0]
    pred_classes = []
    #loop through disc. values to classify
    for ss in range(steps):
        if g1[ss] > g2[ss]:
            pred_classes.append(1)
        elif g1[ss] < g2[ss]:
            pred_classes.append(2)
    return np.array(pred_classes)