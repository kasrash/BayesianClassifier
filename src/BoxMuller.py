import numpy as np
import time

def box_muller(mean, std, no_samples):
    samples = np.random.ranf(size=no_samples)

    x1 = samples[:int(no_samples/2)]
    x2 = samples[int(no_samples/2):]

    y1 = np.sqrt(-2*np.log(x1))*np.cos(2*np.pi*x2)
    y2 = np.sqrt(-2*np.log(x1))*np.sin(2*np.pi*x2)

    y = np.concatenate((y1,y2), axis=0)

    return mean + std*y