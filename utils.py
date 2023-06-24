import numpy as np
import math

def normalize(sample):
    MIN_H = sample.min()
    MAX_H = sample.max()
    return (sample - MIN_H)/(MAX_H-MIN_H)

def calculatePSNR(img1, img2, border=0 ,data_min=0.0 ,data_max=1.0 ):
    
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[2:]

    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10((data_max - data_min)/ math.sqrt(mse))